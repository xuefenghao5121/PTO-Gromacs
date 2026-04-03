/*
 * GROMACS PTO for x86 AVX/AVX2
 * 
 * Tile划分实现 + 算子融合核心（x86 版本）
 * 
 * 移植自 ARM 版本 (gromacs_pto_tiling.c)，适配 x86 缓存层次
 * 
 * 功能:
 * - 空间填充曲线(Tile)划分设计
 * - 基于坐标空间分块，保持空间局部性
 * - 自适应Tile大小（适配 x86 L1/L2/L3 缓存）
 * - 算子融合：中间结果保留在寄存器，消除内存写回
 */

#include "gromacs_pto_x86.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

/* 内部常量 */
#define GMX_PTO_MIN_TILE_SIZE 16
#define GMX_PTO_MAX_TILE_SIZE 1024
#define GMX_PTO_DEFAULT_TILE_SIZE 64
#define GMX_PTO_DEFAULT_L2_CACHE_KB 256

/*
 * 初始化配置为默认值
 */
void gmx_pto_config_x86_init(gmx_pto_config_x86_t *config) {
    config->tile_size_atoms = GMX_PTO_DEFAULT_TILE_SIZE;
    config->tile_size_cache_kb = GMX_PTO_DEFAULT_L2_CACHE_KB;
    config->enable_avx = true;
    config->enable_avx2 = true;
    config->enable_fusion = true;
    config->verbose = false;
}

/*
 * 自动计算最优Tile大小
 * 
 * 估算公式:
 *   每个原子需要存储: 坐标(3 floats) + 力(3 floats) = 24 bytes
 *   加上原子类型参数和邻居索引，约 32 bytes/原子
 *   Tile需要适应L2缓存大小
 * 
 * x86 缓存层次：
 *   L1: 32 KB (每核心)
 *   L2: 256 KB - 1 MB (每核心/每簇)
 *   L3: 8-64 MB (共享)
 */
int gmx_pto_auto_tile_size_x86(int total_atoms, int cache_size_kb) {
    /* 每个原子约32字节数据 */
    int atoms_per_cache = (cache_size_kb * 1024) / 32;
    
    /* 留点余量给其他数据 */
    atoms_per_cache = (int)(atoms_per_cache * 0.8);
    
    /* 限制范围 */
    if (atoms_per_cache < GMX_PTO_MIN_TILE_SIZE) {
        return GMX_PTO_MIN_TILE_SIZE;
    }
    if (atoms_per_cache > GMX_PTO_MAX_TILE_SIZE) {
        return GMX_PTO_MAX_TILE_SIZE;
    }
    
    /* 向下取整到2的幂，便于向量化 */
    int pow2 = 16;
    while (pow2 * 2 <= atoms_per_cache) {
        pow2 *= 2;
    }
    
    return pow2;
}

/*
 * 计算空间边界
 */
static void compute_bounding_box(const float *coords, int num_atoms,
                                 float min_box[3], float max_box[3]) {
    /* 初始化 */
    for (int d = 0; d < 3; d++) {
        min_box[d] = coords[d];
        max_box[d] = coords[d];
    }
    
    /* 遍历所有原子找边界 */
    for (int i = 1; i < num_atoms; i++) {
        for (int d = 0; d < 3; d++) {
            float c = coords[i * 3 + d];
            if (c < min_box[d]) min_box[d] = c;
            if (c > max_box[d]) max_box[d] = c;
        }
    }
}

/*
 * 创建Tile划分
 * 
 * 当前实现: 简化的空间排序分块
 * 后续可优化为完整Hilbert空间填充曲线，进一步提高局部性
 */
int gmx_pto_create_tiling_x86(int total_atoms, const float *coords,
                               const gmx_pto_config_x86_t *config,
                               gmx_pto_nonbonded_context_x86_t *context) {
    if (total_atoms <= 0 || coords == NULL || context == NULL) {
        return -1;
    }
    
    /* 初始化上下文 */
    memset(context, 0, sizeof(*context));
    memcpy(&context->config, config, sizeof(context->config));
    
    context->num_total_atoms = total_atoms;
    
    /* 确定Tile大小 */
    int tile_size = config->tile_size_atoms;
    if (tile_size <= 0) {
        tile_size = gmx_pto_auto_tile_size_x86(total_atoms, config->tile_size_cache_kb);
    }
    
    /* 计算Tile数量 */
    context->num_tiles = (total_atoms + tile_size - 1) / tile_size;
    
    if (config->verbose) {
        printf("[PTO-x86] Creating tiling: total_atoms=%d, tile_size=%d, num_tiles=%d\n",
               total_atoms, tile_size, context->num_tiles);
    }
    
    /* 分配Tile数组 */
    context->tiles = (gmx_pto_tile_x86_t*)calloc(context->num_tiles, sizeof(gmx_pto_tile_x86_t));
    if (context->tiles == NULL) {
        return -2;
    }
    
    /* 计算空间边界 */
    float min_box[3], max_box[3];
    compute_bounding_box(coords, total_atoms, min_box, max_box);
    
    /* 分配并填充Tile */
    for (int t = 0; t < context->num_tiles; t++) {
        gmx_pto_tile_x86_t *tile = &context->tiles[t];
        tile->tile_id = t;
        tile->start_atom = t * tile_size;
        int end_atom = (t + 1) * tile_size;
        if (end_atom > total_atoms) {
            end_atom = total_atoms;
        }
        tile->num_atoms = end_atom - tile->start_atom;
        
        /* 分配原子索引数组 */
        tile->atom_indices = (int*)malloc(tile->num_atoms * sizeof(int));
        if (tile->atom_indices == NULL) {
            /*  cleanup and return error */
            for (int tt = 0; tt < t; tt++) {
                free(context->tiles[tt].atom_indices);
            }
            free(context->tiles);
            context->tiles = NULL;
            return -4;
        }
        
        /* 计算Tile的空间范围 */
        tile->min_coord[0] = tile->min_coord[1] = tile->min_coord[2] = 1e10f;
        tile->max_coord[0] = tile->max_coord[1] = tile->max_coord[2] = -1e10f;
        
        /* 填充原子索引并计算边界 */
        for (int i = 0; i < tile->num_atoms; i++) {
            int global_idx = tile->start_atom + i;  /* 简化: 直接顺序分块 */
            tile->atom_indices[i] = global_idx;
            
            /* 更新边界 */
            for (int d = 0; d < 3; d++) {
                float c = coords[global_idx * 3 + d];
                if (c < tile->min_coord[d]) tile->min_coord[d] = c;
                if (c > tile->max_coord[d]) tile->max_coord[d] = c;
            }
        }
        
        tile->forces_computed = false;
    }
    
    context->num_neighbor_pairs = 0;
    context->neighbor_pairs = NULL;
    
    /* 检查AVX支持 */
    context->avx_enabled = gmx_pto_check_avx_support();
    context->avx2_enabled = gmx_pto_check_avx2_support();
    
    if (config->verbose) {
        gmx_pto_print_info_x86(context);
    }
    
    return 0;
}

/*
 * 销毁Tile划分
 */
void gmx_pto_destroy_tiling_x86(gmx_pto_nonbonded_context_x86_t *context) {
    if (context == NULL) return;
    
    if (context->tiles != NULL) {
        for (int t = 0; t < context->num_tiles; t++) {
            free(context->tiles[t].atom_indices);
        }
        free(context->tiles);
        context->tiles = NULL;
    }
    
    if (context->neighbor_pairs != NULL) {
        for (int p = 0; p < context->num_neighbor_pairs; p++) {
            free(context->neighbor_pairs[p].pairs);
        }
        free(context->neighbor_pairs);
        context->neighbor_pairs = NULL;
    }
    
    context->num_tiles = 0;
    context->num_neighbor_pairs = 0;
}

/*
 * 检查两个Tile在空间上是否相邻（距离小于cutoff）
 */
static bool tiles_are_neighbors(gmx_pto_tile_x86_t *ta, gmx_pto_tile_x86_t *tb, float cutoff) {
    /* 分离轴测试 - 检查是否重叠或在cutoff距离内 */
    for (int d = 0; d < 3; d++) {
        float min_a = ta->min_coord[d];
        float max_a = ta->max_coord[d];
        float min_b = tb->min_coord[d];
        float max_b = tb->max_coord[d];
        
        /* 计算最短距离 */
        float distance = 0.0f;
        if (max_a < min_b) {
            distance = min_b - max_a;
        } else if (max_b < min_a) {
            distance = min_a - max_b;
        } else {
            /* 重叠，此维度距离为0 */
            distance = 0.0f;
        }
        
        if (distance > cutoff) {
            return false;
        }
    }
    return true;
}

/*
 * 构建Tile邻居对
 */
int gmx_pto_build_neighbor_pairs_x86(gmx_pto_nonbonded_context_x86_t *context,
                                     const float *coords,
                                     float cutoff) {
    if (context == NULL || context->num_tiles == 0) {
        return -1;
    }
    
    /* 计算需要多少邻居对 */
    /* 由于空间局部性，每个Tile大约只和相邻几个Tile交互 */
    int estimated_pairs = context->num_tiles * 13 / 2;  /* 3D网格每个tile平均13 neighbor including self */
    
    context->neighbor_pairs = (gmx_pto_neighbor_pair_x86_t*)calloc(estimated_pairs, 
                                                                   sizeof(gmx_pto_neighbor_pair_x86_t));
    if (context->neighbor_pairs == NULL) {
        return -2;
    }
    
    int pair_count = 0;
    
    /* 遍历所有Tile对 (i <= j 避免重复计算) */
    for (int i = 0; i < context->num_tiles; i++) {
        for (int j = i; j < context->num_tiles; j++) {
            gmx_pto_tile_x86_t *tile_i = &context->tiles[i];
            gmx_pto_tile_x86_t *tile_j = &context->tiles[j];
            
            if (tiles_are_neighbors(tile_i, tile_j, cutoff)) {
                /* 这个Tile对需要计算相互作用 */
                gmx_pto_neighbor_pair_x86_t *pair = &context->neighbor_pairs[pair_count];
                pair->tile_i = i;
                pair->tile_j = j;
                
                /* 预计算原子对索引 - 在完整实现中，这会更优化 */
                int np = tile_i->num_atoms * tile_j->num_atoms;
                if (i == j) {
                    np = np / 2;  /* 上三角，避免重复 */
                }
                
                pair->num_pairs = np;
                pair->pairs = (int*)malloc(np * 2 * sizeof(int));
                if (pair->pairs == NULL) {
                    /* 清理已分配的 */
                    for (int p = 0; p < pair_count; p++) {
                        free(context->neighbor_pairs[p].pairs);
                    }
                    free(context->neighbor_pairs);
                    context->neighbor_pairs = NULL;
                    return -3;
                }
                
                /* 填充原子对索引 */
                int idx = 0;
                for (int li = 0; li < tile_i->num_atoms; li++) {
                    int start_j = (i == j) ? (li + 1) : 0;
                    for (int lj = start_j; lj < tile_j->num_atoms; lj++) {
                        pair->pairs[idx * 2 + 0] = li;
                        pair->pairs[idx * 2 + 1] = lj;
                        idx++;
                    }
                }
                
                pair_count++;
                if (pair_count >= estimated_pairs) {
                    break;
                }
            }
        }
    }
    
    context->num_neighbor_pairs = pair_count;
    
    if (context->config.verbose) {
        printf("[PTO-x86] Built neighbor pairs: %d pairs for %d tiles\n",
               pair_count, context->num_tiles);
    }
    
    return 0;
}

/*
 * 获取AVX向量宽度（float个数）
 */
int gmx_pto_get_avx_vector_width(void) {
#ifdef __AVX__
    return 8;  /* 256-bit = 8 floats */
#else
    return 4;  /* SSE: 128-bit = 4 floats */
#endif
}

/*
 * 检查CPU是否支持AVX
 */
bool gmx_pto_check_avx_support(void) {
#ifdef __AVX__
    return true;
#else
    return false;
#endif
}

/*
 * 检查CPU是否支持AVX2
 */
bool gmx_pto_check_avx2_support(void) {
#ifdef __AVX2__
    return true;
#else
    return false;
#endif
}

/*
 * 打印信息
 */
void gmx_pto_print_info_x86(const gmx_pto_nonbonded_context_x86_t *context) {
    printf("=== GROMACS PTO x86 AVX/AVX2 Information ===\n");
    printf("Version: %d.%d.%d\n",
           GROMACS_PTO_X86_VERSION_MAJOR,
           GROMACS_PTO_X86_VERSION_MINOR,
           GROMACS_PTO_X86_VERSION_PATCH);
    printf("Total atoms: %d\n", context->num_total_atoms);
    printf("Number of tiles: %d\n", context->num_tiles);
    printf("Number of neighbor pairs: %d\n", context->num_neighbor_pairs);
    printf("Tile size (atoms): %d\n", context->config.tile_size_atoms);
    printf("AVX enabled: %d\n", context->avx_enabled);
    if (context->avx_enabled) {
        printf("  AVX vector width: %d floats\n", gmx_pto_get_avx_vector_width());
    }
    printf("AVX2 enabled: %d\n", context->avx2_enabled);
    printf("Fusion enabled: %d\n", context->config.enable_fusion);
    printf("============================================\n");
}

/* ========================================================================
 *  算子融合核心实现 - 消除中间内存写回
 * ======================================================================== */

/*
 * 单个Tile的融合计算
 * 
 * 这里是 PTO 的核心：将整个非键相互作用计算流程融合到单个函数中
 * 
 * 传统实现（无融合）：
 *   1. 加载坐标 → 写入临时数组
 *   2. 计算距离 → 写入临时数组
 *   3. 计算力 → 写入临时数组
 *   4. 累加力 → 写回最终数组
 * 
 * PTO 融合实现：
 *   1-4. 全部在寄存器中完成，最后一次性写回
 */
void gmx_pto_nonbonded_compute_tile_x86(gmx_pto_nonbonded_context_x86_t *context,
                                        gmx_pto_atom_data_x86_t *atom_data,
                                        int tile_idx) {
    if (context == NULL || atom_data == NULL || tile_idx >= context->num_tiles) {
        return;
    }
    
    gmx_pto_tile_x86_t *tile = &context->tiles[tile_idx];
    
    /* 找到所有包含此Tile的邻居对并计算 */
    for (int p = 0; p < context->num_neighbor_pairs; p++) {
        gmx_pto_neighbor_pair_x86_t *pair = &context->neighbor_pairs[p];
        
        if (pair->tile_i == tile_idx || pair->tile_j == tile_idx) {
            gmx_pto_tile_x86_t *tile_i = &context->tiles[pair->tile_i];
            gmx_pto_tile_x86_t *tile_j = &context->tiles[pair->tile_j];
            
            /* 计算相互作用 - 使用 AVX 向量化版本 */
            gmx_pto_avx_compute_pair(context, atom_data, tile_i, tile_j);
        }
    }
    
    tile->forces_computed = true;
}

/*
 * 全融合非键计算 - 整个流程融合为单个函数调用
 * 
 * 这就是PTO全流程算子融合:
 * - 原来需要多次函数调用和多次内存读写
 * - 现在一次完成，中间结果保存在寄存器
 */
int gmx_pto_nonbonded_compute_fused_x86(gmx_pto_nonbonded_context_x86_t *context,
                                         gmx_pto_atom_data_x86_t *atom_data) {
    if (context == NULL || atom_data == NULL || context->tiles == NULL) {
        return -1;
    }
    
    if (context->num_neighbor_pairs == 0) {
        /* 需要先调用gmx_pto_build_neighbor_pairs_x86 */
        return -2;
    }
    
    /* 重置计算标志 */
    for (int t = 0; t < context->num_tiles; t++) {
        context->tiles[t].forces_computed = false;
    }
    
    /* 并行遍历所有Tile计算
     * 在实际GROMACS集成中，这里会用OpenMP并行
     * 每个CPU核心负责一个或多个Tile
     */
    #pragma omp parallel for
    for (int t = 0; t < context->num_tiles; t++) {
        gmx_pto_nonbonded_compute_tile_x86(context, atom_data, t);
    }
    
    return 0;
}

/* ========================================================================
 *  兼容 ARM 版本的工具函数实现
 * ======================================================================== */

/*
 * 注意：以下函数与 ARM 版本 (gromacs_pto_tiling.c) 保持接口一致
 * 实现细节根据 x86 平台特性进行了适配
 */

/*
 * 检查Tile是否能放入指定大小缓存
 */
int pto_check_tile_fits_in_cache_x86(int tile_size, int cache_size_kb) {
    /* Each atom needs coords[3] + forces[3] = 6 floats = 24 bytes */
    size_t required_bytes = (size_t)tile_size * 24;
    /* Add overhead for atom indices and other data */
    required_bytes += (size_t)tile_size * sizeof(int);
    /* Use at most 75% of cache to leave room for other data */
    size_t cache_bytes = (size_t)cache_size_kb * 1024;
    size_t max_allowed = (size_t)(cache_bytes * 0.75);
    return (required_bytes <= max_allowed) ? 1 : 0;
}

/*
 * 最小图像约定 - 处理周期性边界条件
 */
void pto_minimum_image_x86(float *dx, float box_length, float box_half) {
    while (*dx > box_half) {
        *dx -= box_length;
    }
    while (*dx < -box_half) {
        *dx += box_length;
    }
}

/*
 * 单个原子对的非键相互作用计算，对称力累加保证牛顿第三定律精确成立
 */
void pto_nonbonded_pair_compute_x86(float coords_i[3], float coords_j[3],
                                     float force_i[3], float force_j[3],
                                     float sigma, float epsilon) {
    /* Calculate distance */
    float dx = coords_j[0] - coords_i[0];
    float dy = coords_j[1] - coords_i[1];
    float dz = coords_j[2] - coords_i[2];
    float rsq = dx*dx + dy*dy + dz*dz;
    
    if (rsq < 1e-12f) {
        /* 避免除以零 */
        return;
    }
    
    float r = sqrtf(rsq);
    
    /* LJ force calculation - symmetric accumulation */
    float sigma_over_r = sigma / r;
    float sigma_over_r6 = sigma_over_r * sigma_over_r;
    sigma_over_r6 = sigma_over_r6 * sigma_over_r6 * sigma_over_r6;
    float dVdr = 12.0f * epsilon * (sigma_over_r6 * sigma_over_r6 - 0.5f * sigma_over_r6) / r;
    
    /* Calculate force component - compute once, apply symmetrically */
    float fx = dVdr * dx / r;
    float fy = dVdr * dy / r;
    float fz = dVdr * dz / r;
    
    /* Symmetric accumulation: i gets +f, j gets -f
     * This guarantees F_i + F_j = 0 exactly in floating point
     */
    force_i[0] += fx;
    force_i[1] += fy;
    force_i[2] += fz;
    force_j[0] -= fx;
    force_j[1] -= fy;
    force_j[2] -= fz;
}

/*
 * 计算LJ能量（参考实现）
 */
float pto_lj_energy_x86(float r, float sigma, float epsilon) {
    if (r <= 0.0f) {
        return 0.0f;
    }
    float sigma_over_r = sigma / r;
    float sigma_over_r6 = sigma_over_r * sigma_over_r;
    sigma_over_r6 = sigma_over_r6 * sigma_over_r6 * sigma_over_r6;
    return 4.0f * epsilon * (sigma_over_r6 * sigma_over_r6 - sigma_over_r6);
}
