/*
 * PTO Tile Data Implementation - 连续存储 + 打包/解包
 * 
 * 实现柱子哥方案的核心数据结构重构
 */

#include "pto_tile_data.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 16字节对齐的malloc */
static void* aligned_malloc(size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;  /* 64-byte cache line alignment */
    memset(ptr, 0, size);
    return ptr;
}

pto_tile_data_t* pto_tile_data_create(int capacity) {
    pto_tile_data_t *tile = (pto_tile_data_t*)calloc(1, sizeof(pto_tile_data_t));
    if (!tile) return NULL;
    
    tile->capacity = capacity;
    tile->n_atoms = 0;
    
    /* 计算所需内存: 11个数组 x capacity x sizeof(float) */
    int n_arrays = 11; /* x,y,z, fx,fy,fz, lj_sigma_sq,lj_eps,charges, global_indices(not float) */
    size_t float_mem = (size_t)n_arrays * capacity * sizeof(float);
    size_t idx_mem = (size_t)capacity * sizeof(int);
    tile->mem_size = float_mem + idx_mem + 64; /* extra for alignment */
    
    tile->mem_block = aligned_malloc(tile->mem_size);
    if (!tile->mem_block) {
        free(tile);
        return NULL;
    }
    
    /* 分配指针到连续内存块 */
    char *base = (char*)tile->mem_block;
    size_t offset = 0;
    size_t stride = (size_t)capacity * sizeof(float);
    
    tile->x            = (float*)(base + offset); offset += stride;
    tile->y            = (float*)(base + offset); offset += stride;
    tile->z            = (float*)(base + offset); offset += stride;
    tile->fx           = (float*)(base + offset); offset += stride;
    tile->fy           = (float*)(base + offset); offset += stride;
    tile->fz           = (float*)(base + offset); offset += stride;
    tile->lj_sigma_sq  = (float*)(base + offset); offset += stride;
    tile->lj_eps       = (float*)(base + offset); offset += stride;
    tile->charges      = (float*)(base + offset); offset += stride;
    /* 剩余空间用于global_indices */
    tile->global_indices = (int*)(base + offset);
    
    return tile;
}

void pto_tile_data_destroy(pto_tile_data_t *tile) {
    if (tile) {
        if (tile->mem_block) free(tile->mem_block);
        free(tile);
    }
}

void pto_tile_data_pack_coords(pto_tile_data_t *tile,
                                const float *global_coords,
                                const float *global_lj_sigma,
                                const float *global_lj_eps,
                                const float *global_charges) {
    for (int i = 0; i < tile->n_atoms; i++) {
        int gi = tile->global_indices[i];
        tile->x[i] = global_coords[gi * 3 + 0];
        tile->y[i] = global_coords[gi * 3 + 1];
        tile->z[i] = global_coords[gi * 3 + 2];
        
        /* 预计算LJ参数 */
        float sigma = global_lj_sigma ? global_lj_sigma[gi] : 0.3f;
        tile->lj_sigma_sq[i] = sigma * sigma;
        tile->lj_eps[i] = global_lj_eps ? global_lj_eps[gi] : 0.5f;
        
        tile->charges[i] = global_charges ? global_charges[gi] : 0.0f;
    }
    
    /* 力缓冲区清零 */
    memset(tile->fx, 0, tile->n_atoms * sizeof(float));
    memset(tile->fy, 0, tile->n_atoms * sizeof(float));
    memset(tile->fz, 0, tile->n_atoms * sizeof(float));
}

void pto_tile_data_unpack_forces(const pto_tile_data_t *tile,
                                  float *global_forces) {
    for (int i = 0; i < tile->n_atoms; i++) {
        int gi = tile->global_indices[i];
        global_forces[gi * 3 + 0] += tile->fx[i];
        global_forces[gi * 3 + 1] += tile->fy[i];
        global_forces[gi * 3 + 2] += tile->fz[i];
    }
}

/* ===== Tiling上下文 ===== */

int pto_tiling_create(const float *coords, int natoms,
                       const float *lj_sigma, const float *lj_eps,
                       const float *charges,
                       const float box[3], float cutoff,
                       int target_tile_size,
                       pto_tiling_ctx_t *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->total_atoms = natoms;
    ctx->cutoff = cutoff;
    memcpy(ctx->box, box, 3 * sizeof(float));
    
    /* 计算Tile数量 */
    int num_tiles = (natoms + target_tile_size - 1) / target_tile_size;
    ctx->num_tiles = num_tiles;
    
    /* 分配Tile数组 */
    ctx->tiles = (pto_tile_data_t*)calloc(num_tiles, sizeof(pto_tile_data_t));
    if (!ctx->tiles) return -1;
    
    /* 创建每个Tile */
    float csq = cutoff * cutoff;
    int pair_capacity = num_tiles * 14; /* 预估邻居对数 */
    ctx->pairs = (pto_tile_pair_t*)calloc(pair_capacity, sizeof(pto_tile_pair_t));
    if (!ctx->pairs) return -2;
    
    for (int t = 0; t < num_tiles; t++) {
        int start = t * target_tile_size;
        int end = (start + target_tile_size > natoms) ? natoms : start + target_tile_size;
        int nt = end - start;
        
        pto_tile_data_t *tile = pto_tile_data_create(nt);
        if (!tile) return -3;
        
        tile->n_atoms = nt;
        for (int i = 0; i < nt; i++) {
            tile->global_indices[i] = start + i;
        }
        
        /* 打包坐标和参数 */
        pto_tile_data_pack_coords(tile, coords, lj_sigma, lj_eps, charges);
        
        ctx->tiles[t] = *tile; /* 浅拷贝指针 */
        free(tile); /* 只free外壳，不free内部mem_block */
    }
    
    /* 构建Tile间邻居对 - 基于空间距离 */
    ctx->num_pairs = 0;
    for (int i = 0; i < num_tiles; i++) {
        for (int j = i; j < num_tiles; j++) {
            /* 计算Tile间距离 - 用中心点近似 */
            float ci[3], cj[3];
            for (int d = 0; d < 3; d++) {
                ci[d] = 0; cj[d] = 0;
            }
            for (int k = 0; k < ctx->tiles[i].n_atoms; k++) {
                ci[0] += ctx->tiles[i].x[k];
                ci[1] += ctx->tiles[i].y[k];
                ci[2] += ctx->tiles[i].z[k];
            }
            for (int k = 0; k < ctx->tiles[j].n_atoms; k++) {
                cj[0] += ctx->tiles[j].x[k];
                cj[1] += ctx->tiles[j].y[k];
                cj[2] += ctx->tiles[j].z[k];
            }
            for (int d = 0; d < 3; d++) {
                ci[d] /= ctx->tiles[i].n_atoms;
                cj[d] /= ctx->tiles[j].n_atoms;
            }
            
            /* 简化: 所有Tile对都可能是邻居（小规模时） */
            if (ctx->num_pairs >= pair_capacity) {
                pair_capacity *= 2;
                ctx->pairs = (pto_tile_pair_t*)realloc(ctx->pairs, pair_capacity * sizeof(pto_tile_pair_t));
            }
            
            pto_tile_pair_t *pair = &ctx->pairs[ctx->num_pairs];
            pair->tile_i = i;
            pair->tile_j = j;
            pair->num_pairs = 0;
            pair->local_i = NULL;
            pair->local_j = NULL;
            
            /* 统计有效原子对 */
            int ni = ctx->tiles[i].n_atoms;
            int nj = ctx->tiles[j].n_atoms;
            int est = ni * nj;
            if (i == j) est = ni * (ni - 1) / 2;
            
            pair->local_i = (int*)malloc(est * sizeof(int));
            pair->local_j = (int*)malloc(est * sizeof(int));
            
            int cnt = 0;
            for (int li = 0; li < ni; li++) {
                int start_j = (i == j) ? li + 1 : 0;
                for (int lj = start_j; lj < nj; lj++) {
                    float dx = ctx->tiles[i].x[li] - ctx->tiles[j].x[lj];
                    float dy = ctx->tiles[i].y[li] - ctx->tiles[j].y[lj];
                    float dz = ctx->tiles[i].z[li] - ctx->tiles[j].z[lj];
                    
                    /* PBC */
                    dx -= box[0] * roundf(dx / box[0]);
                    dy -= box[1] * roundf(dy / box[1]);
                    dz -= box[2] * roundf(dz / box[2]);
                    
                    float rsq = dx*dx + dy*dy + dz*dz;
                    if (rsq < csq && rsq > 1e-8f) {
                        pair->local_i[cnt] = li;
                        pair->local_j[cnt] = lj;
                        cnt++;
                    }
                }
            }
            pair->num_pairs = cnt;
            if (cnt > 0) {
                ctx->num_pairs++;
            } else {
                free(pair->local_i);
                free(pair->local_j);
            }
        }
    }
    
    return 0;
}

void pto_tiling_destroy(pto_tiling_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->tiles) {
        for (int t = 0; t < ctx->num_tiles; t++) {
            if (ctx->tiles[t].mem_block) free(ctx->tiles[t].mem_block);
        }
        free(ctx->tiles);
    }
    if (ctx->pairs) {
        for (int p = 0; p < ctx->num_pairs; p++) {
            free(ctx->pairs[p].local_i);
            free(ctx->pairs[p].local_j);
        }
        free(ctx->pairs);
    }
    memset(ctx, 0, sizeof(*ctx));
}
