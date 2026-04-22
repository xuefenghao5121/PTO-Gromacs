/*
 * PTO-GROMACS 端到端性能对比测试
 * 
 * 三组对照:
 *   1. 标量基线 (无向量化)
 *   2. 传统 SVE 向量化 (无 PTO 融合/Tile 排序)
 *   3. PTO-SVE 算子融合 (SoA + Tile 排序 + 连续加载)
 *
 * 编译 (鲲鹏930):
 *   gcc -O3 -march=armv9-a+sve+sve2 -msve-vector-bits=256 -ffast-math -fopenmp \
 *       pto_gromacs_e2e_benchmark.c -o pto_e2e_benchmark -lm
 *
 * 运行:
 *   OMP_NUM_THREADS=16 ./pto_e2e_benchmark em_medium.gro 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_sve.h>
#include <omp.h>

/* ============ GROMACS .gro 文件解析 ============ */

typedef struct {
    int natoms;
    float *x;       /* x[0..natoms*3-1], nm */
    float box[3];   /* box dimensions nm */
} GroData;

static GroData* read_gro(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return NULL; }
    
    GroData *gro = calloc(1, sizeof(GroData));
    char line[256];
    
    fgets(line, sizeof(line), fp); /* title */
    fgets(line, sizeof(line), fp); /* atom count */
    gro->natoms = atoi(line);
    gro->x = malloc(gro->natoms * 3 * sizeof(float));
    
    for (int i = 0; i < gro->natoms; i++) {
        fgets(line, sizeof(line), fp);
        char *p = line + 20;
        gro->x[i*3+0] = strtof(p, &p);
        gro->x[i*3+1] = strtof(p, &p);
        gro->x[i*3+2] = strtof(p, &p);
    }
    
    fgets(line, sizeof(line), fp);
    sscanf(line, "%f %f %f", &gro->box[0], &gro->box[1], &gro->box[2]);
    fclose(fp);
    return gro;
}

static void free_gro(GroData *gro) {
    if (gro) { free(gro->x); free(gro); }
}

/* ============ 邻居列表构建 ============ */

typedef struct {
    int *start;     /* start[i] = prefix sum */
    int *jatoms;    /* j-atom indices */
    int *count;     /* count[i] = number of neighbors */
    int total_pairs;
} NeighborList;

static NeighborList* build_neighbor_list(float *x, int natoms, float box[3], float cutoff) {
    NeighborList *nl = calloc(1, sizeof(NeighborList));
    nl->start = malloc(natoms * sizeof(int));
    nl->count = calloc(natoms, sizeof(int));
    
    float cutoff_sq = cutoff * cutoff;
    
    /* Count pass */
    for (int i = 0; i < natoms; i++) {
        for (int j = i + 1; j < natoms; j++) {
            float dx = x[i*3+0] - x[j*3+0];
            float dy = x[i*3+1] - x[j*3+1];
            float dz = x[i*3+2] - x[j*3+2];
            dx -= box[0] * roundf(dx / box[0]);
            dy -= box[1] * roundf(dy / box[1]);
            dz -= box[2] * roundf(dz / box[2]);
            if (dx*dx + dy*dy + dz*dz < cutoff_sq) nl->count[i]++;
        }
    }
    
    /* Prefix sum */
    nl->start[0] = 0;
    for (int i = 1; i < natoms; i++)
        nl->start[i] = nl->start[i-1] + nl->count[i-1];
    nl->total_pairs = nl->start[natoms-1] + nl->count[natoms-1];
    nl->jatoms = malloc(nl->total_pairs * sizeof(int));
    
    /* Fill pass */
    int *pos = calloc(natoms, sizeof(int));
    for (int i = 0; i < natoms; i++) {
        for (int j = i + 1; j < natoms; j++) {
            float dx = x[i*3+0] - x[j*3+0];
            float dy = x[i*3+1] - x[j*3+1];
            float dz = x[i*3+2] - x[j*3+2];
            dx -= box[0] * roundf(dx / box[0]);
            dy -= box[1] * roundf(dy / box[1]);
            dz -= box[2] * roundf(dz / box[2]);
            if (dx*dx + dy*dy + dz*dz < cutoff_sq)
                nl->jatoms[nl->start[i] + pos[i]++] = j;
        }
    }
    free(pos);
    return nl;
}

static void free_neighbor_list(NeighborList *nl) {
    if (nl) { free(nl->start); free(nl->jatoms); free(nl->count); free(nl); }
}

/* ====================================================================
 * 组 1: 标量基线 - 无向量化, per-thread 缓冲区 (无 atomic)
 * ==================================================================== */
static double compute_forces_scalar(float *x, float *f, int natoms, float box[3],
                                    NeighborList *nl, float cutoff) {
    float cutoff_sq = cutoff * cutoff;
    float eps = 0.5f, sigma_sq = 0.09f; /* 0.3^2 */
    
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    memset(f, 0, natoms * 3 * sizeof(float));
    
    #pragma omp parallel
    {
        /* Per-thread force buffer - 消除 atomic */
        float *lfx = calloc(natoms, sizeof(float));
        float *lfy = calloc(natoms, sizeof(float));
        float *lfz = calloc(natoms, sizeof(float));
        
        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < natoms; i++) {
            float xi = x[i*3+0], yi = x[i*3+1], zi = x[i*3+2];
            
            for (int n = 0; n < nl->count[i]; n++) {
                int j = nl->jatoms[nl->start[i] + n];
                float dx = xi - x[j*3+0];
                float dy = yi - x[j*3+1];
                float dz = zi - x[j*3+2];
                dx -= box[0] * roundf(dx / box[0]);
                dy -= box[1] * roundf(dy / box[1]);
                dz -= box[2] * roundf(dz / box[2]);
                
                float rsq = dx*dx + dy*dy + dz*dz;
                if (rsq < cutoff_sq && rsq > 1e-8f) {
                    float ir = 1.0f / rsq;
                    float s2 = sigma_sq * ir;
                    float s6 = s2*s2*s2;
                    float s12 = s6*s6;
                    float fr = 24.0f * eps * (2.0f*s12 - s6) * ir;
                    float fx = fr*dx, fy = fr*dy, fz = fr*dz;
                    lfx[i] += fx;  lfy[i] += fy;  lfz[i] += fz;
                    lfx[j] -= fx;  lfy[j] -= fy;  lfz[j] -= fz;
                }
            }
        }
        
        /* 合并到全局力数组 */
        #pragma omp critical
        for (int k = 0; k < natoms; k++) {
            f[k*3+0] += lfx[k];
            f[k*3+1] += lfy[k];
            f[k*3+2] += lfz[k];
        }
        
        free(lfx); free(lfy); free(lfz);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* ====================================================================
 * 组 2: 传统 SVE 向量化 - 有向量化, 无 PTO 融合/Tile 排序
 * 
 * 这是对照组: 用 SVE 向量化做力计算, 但不使用:
 *   - SoA 数据布局 (仍用 AoS stride-3 访问)
 *   - Tile-sorted 邻居列表 (邻居顺序未优化)
 *   - 连续加载路径 (全部走 gather fallback)
 * 
 * 目的: 分离 SVE 向量化本身带来的收益
 * ==================================================================== */
static double compute_forces_sve_naive(float *x, float *f, int natoms, float box[3],
                                        NeighborList *nl, float cutoff) {
    float cutoff_sq = cutoff * cutoff;
    float eps = 0.5f, sigma_sq = 0.09f;
    
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    memset(f, 0, natoms * 3 * sizeof(float));
    
    #pragma omp parallel
    {
        float *lfx = calloc(natoms, sizeof(float));
        float *lfy = calloc(natoms, sizeof(float));
        float *lfz = calloc(natoms, sizeof(float));
        
        svbool_t all_p = svptrue_b32();
        int vl = (int)svcntw();
        
        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < natoms; i++) {
            float xi = x[i*3+0], yi = x[i*3+1], zi = x[i*3+2];
            svfloat32_t fx_acc = svdup_f32(0);
            svfloat32_t fy_acc = svdup_f32(0);
            svfloat32_t fz_acc = svdup_f32(0);
            
            int nn = nl->count[i];
            const int *j_atoms = &nl->jatoms[nl->start[i]];
            
            /* 按 SVE 向量宽度分块处理 j 原子 - 但不做 Tile 排序, 全部 gather */
            for (int k = 0; k < nn; k += vl) {
                int rem = nn - k;
                svbool_t pg = svwhilelt_b32(0, rem);
                
                /* ★ 传统方式: 标量 gather 到栈缓冲区 (stride-3 AoS) */
                float jx_buf[16], jy_buf[16], jz_buf[16];
                int jj_buf[16];
                int cnt = (rem < vl) ? rem : vl;
                for (int m = 0; m < cnt; m++) {
                    int j = j_atoms[k + m];
                    jj_buf[m] = j;
                    jx_buf[m] = x[j*3+0];  /* stride-3 gather */
                    jy_buf[m] = x[j*3+1];
                    jz_buf[m] = x[j*3+2];
                }
                
                svfloat32_t xj_v = svld1_f32(pg, jx_buf);
                svfloat32_t yj_v = svld1_f32(pg, jy_buf);
                svfloat32_t zj_v = svld1_f32(pg, jz_buf);
                
                /* 距离计算 (在寄存器中) */
                svfloat32_t dx = svsub_f32_x(pg, svdup_f32(xi), xj_v);
                svfloat32_t dy = svsub_f32_x(pg, svdup_f32(yi), yj_v);
                svfloat32_t dz = svsub_f32_x(pg, svdup_f32(zi), zj_v);
                
                /* PBC */
                dx = svsub_f32_x(pg, dx, svmul_f32_x(pg, svdup_f32(box[0]),
                    svrinta_f32_x(pg, svdiv_f32_x(pg, dx, svdup_f32(box[0])))));
                dy = svsub_f32_x(pg, dy, svmul_f32_x(pg, svdup_f32(box[1]),
                    svrinta_f32_x(pg, svdiv_f32_x(pg, dy, svdup_f32(box[1])))));
                dz = svsub_f32_x(pg, dz, svmul_f32_x(pg, svdup_f32(box[2]),
                    svrinta_f32_x(pg, svdiv_f32_x(pg, dz, svdup_f32(box[2])))));
                
                svfloat32_t rsq = svadd_f32_x(pg, svadd_f32_x(pg,
                    svmul_f32_x(pg, dx, dx), svmul_f32_x(pg, dy, dy)),
                    svmul_f32_x(pg, dz, dz));
                
                svbool_t valid = svand_b_z(pg,
                    svcmplt_f32(pg, rsq, svdup_f32(cutoff_sq)),
                    svcmpgt_f32(pg, rsq, svdup_f32(1e-8f)));
                
                if (svptest_any(all_p, valid)) {
                    svfloat32_t ir = svdiv_f32_z(valid, svdup_f32(1.0f), rsq);
                    svfloat32_t s2 = svmul_f32_z(valid, svdup_f32(sigma_sq), ir);
                    svfloat32_t s6 = svmul_f32_z(valid, svmul_f32_z(valid, s2, s2), s2);
                    svfloat32_t s12 = svmul_f32_z(valid, s6, s6);
                    svfloat32_t fr = svmul_f32_z(valid, svdup_f32(24.0f * eps),
                        svmul_f32_z(valid, svsub_f32_z(valid,
                            svmul_f32_z(valid, svdup_f32(2.0f), s12), s6), ir));
                    
                    svfloat32_t fx_v = svmul_f32_z(valid, fr, dx);
                    svfloat32_t fy_v = svmul_f32_z(valid, fr, dy);
                    svfloat32_t fz_v = svmul_f32_z(valid, fr, dz);
                    
                    fx_acc = svadd_f32_m(valid, fx_acc, fx_v);
                    fy_acc = svadd_f32_m(valid, fy_acc, fy_v);
                    fz_acc = svadd_f32_m(valid, fz_acc, fz_v);
                    
                    /* j 原子力 - 标量 scatter */
                    float fxo[16], fyo[16], fzo[16];
                    svst1_f32(pg, fxo, fx_v);
                    svst1_f32(pg, fyo, fy_v);
                    svst1_f32(pg, fzo, fz_v);
                    for (int m = 0; m < cnt; m++) {
                        int j = jj_buf[m];
                        lfx[j] -= fxo[m];
                        lfy[j] -= fyo[m];
                        lfz[j] -= fzo[m];
                    }
                }
            }
            
            lfx[i] += svaddv_f32(all_p, fx_acc);
            lfy[i] += svaddv_f32(all_p, fy_acc);
            lfz[i] += svaddv_f32(all_p, fz_acc);
        }
        
        #pragma omp critical
        for (int k = 0; k < natoms; k++) {
            f[k*3+0] += lfx[k];
            f[k*3+1] += lfy[k];
            f[k*3+2] += lfz[k];
        }
        
        free(lfx); free(lfy); free(lfz);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* ====================================================================
 * 组 3: PTO-SVE 算子融合 - SoA + Tile-sorted + 连续加载
 * 
 * 关键 PTO 融合优化:
 *   - SoA 数据布局 (消除 stride-3)
 *   - Tile-sorted 邻居列表 (j 索引按 tile 分组排序)
 *   - svld1 连续加载路径 (同 tile 内 j 索引连续, 无 gather)
 *   - 累加器保持在 SVE 向量寄存器 (中间结果不写内存)
 * ==================================================================== */

/* PTO 上下文 */
typedef struct {
    int n;
    int num_tiles;
    int tile_size;
    
    /* SoA 坐标 */
    float *sx, *sy, *sz;
    
    /* Per-thread SoA 力缓冲区 */
    float **thread_lfx, **thread_lfy, **thread_lfz;
    
    /* Tile-sorted 邻居列表 */
    int **tile_run_start;  /* [n][num_tiles] */
    int **tile_run_count;  /* [n][num_tiles] */
    int  *sorted_jatoms;   /* sorted j indices */
} PTOCtx;

static PTOCtx* pto_init(const float *aos_coords, int n, NeighborList *nl, int tile_size) {
    PTOCtx *ctx = calloc(1, sizeof(PTOCtx));
    ctx->n = n;
    ctx->tile_size = tile_size;
    ctx->num_tiles = (n + tile_size - 1) / tile_size;
    
    /* SoA */
    ctx->sx = malloc(n * sizeof(float));
    ctx->sy = malloc(n * sizeof(float));
    ctx->sz = malloc(n * sizeof(float));
    
    /* Per-thread buffers */
    int nt = omp_get_max_threads();
    ctx->thread_lfx = malloc(nt * sizeof(float*));
    ctx->thread_lfy = malloc(nt * sizeof(float*));
    ctx->thread_lfz = malloc(nt * sizeof(float*));
    for (int t = 0; t < nt; t++) {
        ctx->thread_lfx[t] = calloc(n, sizeof(float));
        ctx->thread_lfy[t] = calloc(n, sizeof(float));
        ctx->thread_lfz[t] = calloc(n, sizeof(float));
    }
    
    /* Tile-sorted neighbor list */
    ctx->tile_run_start = malloc(n * sizeof(int*));
    ctx->tile_run_count = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        ctx->tile_run_start[i] = calloc(ctx->num_tiles, sizeof(int));
        ctx->tile_run_count[i] = calloc(ctx->num_tiles, sizeof(int));
    }
    ctx->sorted_jatoms = malloc(nl->total_pairs * sizeof(int));
    
    /* Sort j atoms by tile */
    int *tile_tmp = malloc(ctx->num_tiles * sizeof(int));
    for (int i = 0; i < n; i++) {
        int ni = nl->count[i];
        int *js = &nl->jatoms[nl->start[i]];
        
        memset(tile_tmp, 0, ctx->num_tiles * sizeof(int));
        for (int k = 0; k < ni; k++) {
            int ti = js[k] / tile_size;
            if (ti >= ctx->num_tiles) ti = ctx->num_tiles - 1;
            tile_tmp[ti]++;
        }
        
        int offset = 0;
        for (int ti = 0; ti < ctx->num_tiles; ti++) {
            ctx->tile_run_count[i][ti] = tile_tmp[ti];
            ctx->tile_run_start[i][ti] = offset;
            offset += tile_tmp[ti];
        }
        
        int *pos = calloc(ctx->num_tiles, sizeof(int));
        for (int k = 0; k < ni; k++) {
            int j = js[k];
            int ti = j / tile_size;
            if (ti >= ctx->num_tiles) ti = ctx->num_tiles - 1;
            ctx->sorted_jatoms[nl->start[i] + ctx->tile_run_start[i][ti] + pos[ti]] = j;
            pos[ti]++;
        }
        free(pos);
    }
    free(tile_tmp);
    
    /* Initial SoA pack */
    for (int i = 0; i < n; i++) {
        ctx->sx[i] = aos_coords[i*3+0];
        ctx->sy[i] = aos_coords[i*3+1];
        ctx->sz[i] = aos_coords[i*3+2];
    }
    
    return ctx;
}

static void pto_destroy(PTOCtx *ctx) {
    free(ctx->sx); free(ctx->sy); free(ctx->sz);
    int nt = omp_get_max_threads();
    for (int t = 0; t < nt; t++) {
        free(ctx->thread_lfx[t]);
        free(ctx->thread_lfy[t]);
        free(ctx->thread_lfz[t]);
    }
    free(ctx->thread_lfx); free(ctx->thread_lfy); free(ctx->thread_lfz);
    for (int i = 0; i < ctx->n; i++) {
        free(ctx->tile_run_start[i]);
        free(ctx->tile_run_count[i]);
    }
    free(ctx->tile_run_start); free(ctx->tile_run_count);
    free(ctx->sorted_jatoms);
    free(ctx);
}

static double compute_forces_pto_sve(PTOCtx *ctx, float *f, int n, float box[3],
                                      NeighborList *nl, float cutoff) {
    float cutoff_sq = cutoff * cutoff;
    float eps = 0.5f, sigma_sq = 0.09f;
    int num_tiles = ctx->num_tiles;
    float *sx = ctx->sx, *sy = ctx->sy, *sz = ctx->sz;
    
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    memset(f, 0, n * 3 * sizeof(float));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *lfx = ctx->thread_lfx[tid];
        float *lfy = ctx->thread_lfy[tid];
        float *lfz = ctx->thread_lfz[tid];
        memset(lfx, 0, n * sizeof(float));
        memset(lfy, 0, n * sizeof(float));
        memset(lfz, 0, n * sizeof(float));
        
        svbool_t all_p = svptrue_b32();
        int vl = (int)svcntw();
        
        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < n; i++) {
            float xi = sx[i], yi = sy[i], zi = sz[i];
            svfloat32_t fx_a = svdup_f32(0), fy_a = svdup_f32(0), fz_a = svdup_f32(0);
            
            for (int ti = 0; ti < num_tiles; ti++) {
                int run_start = ctx->tile_run_start[i][ti];
                int run_count = ctx->tile_run_count[i][ti];
                if (run_count == 0) continue;
                
                /* ★ PTO 优化: 检查同 tile 内 j 索引是否连续 */
                int is_contig = 1;
                for (int m = 1; m < run_count && m < vl; m++) {
                    int j_prev = ctx->sorted_jatoms[nl->start[i] + run_start + m - 1];
                    int j_curr = ctx->sorted_jatoms[nl->start[i] + run_start + m];
                    if (j_curr != j_prev + 1) { is_contig = 0; break; }
                }
                
                for (int k = 0; k < run_count; k += vl) {
                    int rem = run_count - k;
                    svbool_t pg = svwhilelt_b32(0, rem);
                    
                    svfloat32_t dx_v, dy_v, dz_v;
                    
                    if (is_contig) {
                        /* ★ PTO 连续加载路径 - 一条 svld1, 无 gather! */
                        int j0 = ctx->sorted_jatoms[nl->start[i] + run_start + k];
                        svfloat32_t xj = svld1_f32(pg, &sx[j0]);
                        svfloat32_t yj = svld1_f32(pg, &sy[j0]);
                        svfloat32_t zj = svld1_f32(pg, &sz[j0]);
                        dx_v = svsub_f32_x(pg, svdup_f32(xi), xj);
                        dy_v = svsub_f32_x(pg, svdup_f32(yi), yj);
                        dz_v = svsub_f32_x(pg, svdup_f32(zi), zj);
                    } else {
                        /* Fallback: gather */
                        float jx[16], jy[16], jz[16];
                        for (int m = 0; m < rem; m++) {
                            int j = ctx->sorted_jatoms[nl->start[i] + run_start + k + m];
                            jx[m] = xi - sx[j];
                            jy[m] = yi - sy[j];
                            jz[m] = zi - sz[j];
                        }
                        dx_v = svld1_f32(pg, jx);
                        dy_v = svld1_f32(pg, jy);
                        dz_v = svld1_f32(pg, jz);
                    }
                    
                    /* PBC */
                    dx_v = svsub_f32_x(pg, dx_v, svmul_f32_x(pg, svdup_f32(box[0]),
                        svrinta_f32_x(pg, svdiv_f32_x(pg, dx_v, svdup_f32(box[0])))));
                    dy_v = svsub_f32_x(pg, dy_v, svmul_f32_x(pg, svdup_f32(box[1]),
                        svrinta_f32_x(pg, svdiv_f32_x(pg, dy_v, svdup_f32(box[1])))));
                    dz_v = svsub_f32_x(pg, dz_v, svmul_f32_x(pg, svdup_f32(box[2]),
                        svrinta_f32_x(pg, svdiv_f32_x(pg, dz_v, svdup_f32(box[2])))));
                    
                    svfloat32_t rsq = svadd_f32_x(pg, svadd_f32_x(pg,
                        svmul_f32_x(pg, dx_v, dx_v), svmul_f32_x(pg, dy_v, dy_v)),
                        svmul_f32_x(pg, dz_v, dz_v));
                    
                    svbool_t valid = svand_b_z(pg,
                        svcmplt_f32(pg, rsq, svdup_f32(cutoff_sq)),
                        svcmpgt_f32(pg, rsq, svdup_f32(1e-8f)));
                    
                    if (svptest_any(all_p, valid)) {
                        svfloat32_t ir = svdiv_f32_z(valid, svdup_f32(1.0f), rsq);
                        svfloat32_t s2 = svmul_f32_z(valid, svdup_f32(sigma_sq), ir);
                        svfloat32_t s6 = svmul_f32_z(valid, svmul_f32_z(valid, s2, s2), s2);
                        svfloat32_t s12 = svmul_f32_z(valid, s6, s6);
                        svfloat32_t fr = svmul_f32_z(valid, svdup_f32(24.0f * eps),
                            svmul_f32_z(valid, svsub_f32_z(valid,
                                svmul_f32_z(valid, svdup_f32(2.0f), s12), s6), ir));
                        
                        svfloat32_t fx_v = svmul_f32_z(valid, fr, dx_v);
                        svfloat32_t fy_v = svmul_f32_z(valid, fr, dy_v);
                        svfloat32_t fz_v = svmul_f32_z(valid, fr, dz_v);
                        
                        fx_a = svadd_f32_m(valid, fx_a, fx_v);
                        fy_a = svadd_f32_m(valid, fy_a, fy_v);
                        fz_a = svadd_f32_m(valid, fz_a, fz_v);
                        
                        float fxo[16], fyo[16], fzo[16];
                        svst1_f32(pg, fxo, fx_v);
                        svst1_f32(pg, fyo, fy_v);
                        svst1_f32(pg, fzo, fz_v);
                        for (int m = 0; m < rem; m++) {
                            int j = ctx->sorted_jatoms[nl->start[i] + run_start + k + m];
                            lfx[j] -= fxo[m];
                            lfy[j] -= fyo[m];
                            lfz[j] -= fzo[m];
                        }
                    }
                }
            }
            lfx[i] += svaddv_f32(all_p, fx_a);
            lfy[i] += svaddv_f32(all_p, fy_a);
            lfz[i] += svaddv_f32(all_p, fz_a);
        }
        
        #pragma omp critical
        for (int k = 0; k < n; k++) {
            f[k*3+0] += lfx[k];
            f[k*3+1] += lfy[k];
            f[k*3+2] += lfz[k];
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* ============ Main ============ */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <file.gro> [cutoff_nm] [nsteps] [tile_size]\n", argv[0]);
        return 1;
    }
    
    const char *gro_file = argv[1];
    float cutoff = (argc > 2) ? atof(argv[2]) : 1.0f;
    int nsteps = (argc > 3) ? atoi(argv[3]) : 100;
    int tile_size = (argc > 4) ? atoi(argv[4]) : 64;
    
    printf("================================================================\n");
    printf("  PTO-GROMACS 三组对照 Benchmark\n");
    printf("================================================================\n");
    printf("File: %s | Cutoff: %.2f nm | Steps: %d | Tile: %d\n",
           gro_file, cutoff, nsteps, tile_size);
    printf("SVE: %d-bit (%d floats) | Threads: %d\n\n",
           svcntb()*8, svcntw(), omp_get_max_threads());
    
    /* Read coordinates */
    GroData *gro = read_gro(gro_file);
    if (!gro) return 1;
    int n = gro->natoms;
    printf("Atoms: %d | Box: %.3f x %.3f x %.3f nm\n", n, gro->box[0], gro->box[1], gro->box[2]);
    
    /* Build neighbor list */
    printf("\nBuilding neighbor list..."); fflush(stdout);
    NeighborList *nl = build_neighbor_list(gro->x, n, gro->box, cutoff);
    printf(" %d pairs (%.1f nbs/atom)\n\n", nl->total_pairs, 2.0f * nl->total_pairs / n);
    
    /* Init PTO context */
    PTOCtx *ctx = pto_init(gro->x, n, nl, tile_size);
    
    float *f_scalar = malloc(n * 3 * sizeof(float));
    float *f_sve = malloc(n * 3 * sizeof(float));
    float *f_pto = malloc(n * 3 * sizeof(float));
    
    /* ---- 组 1: 标量基线 ---- */
    printf("组1 Scalar:    "); fflush(stdout);
    compute_forces_scalar(gro->x, f_scalar, n, gro->box, nl, cutoff);
    double t_scalar = 0;
    for (int s = 0; s < nsteps; s++)
        t_scalar += compute_forces_scalar(gro->x, f_scalar, n, gro->box, nl, cutoff);
    printf("%.3f ms/step (1.00x)\n", t_scalar / nsteps * 1000);
    
    /* ---- 组 2: 传统 SVE (无 PTO 融合) ---- */
    printf("组2 SVE naive: "); fflush(stdout);
    compute_forces_sve_naive(gro->x, f_sve, n, gro->box, nl, cutoff);
    double t_sve = 0;
    for (int s = 0; s < nsteps; s++)
        t_sve += compute_forces_sve_naive(gro->x, f_sve, n, gro->box, nl, cutoff);
    printf("%.3f ms/step (%.2fx)\n", t_sve / nsteps * 1000, t_scalar / t_sve);
    
    /* ---- 组 3: PTO-SVE (算子融合) ---- */
    printf("组3 PTO-SVE:   "); fflush(stdout);
    compute_forces_pto_sve(ctx, f_pto, n, gro->box, nl, cutoff);
    double t_pto = 0;
    for (int s = 0; s < nsteps; s++)
        t_pto += compute_forces_pto_sve(ctx, f_pto, n, gro->box, nl, cutoff);
    printf("%.3f ms/step (%.2fx)\n", t_pto / nsteps * 1000, t_scalar / t_pto);
    
    /* Force validation */
    float max_diff_sve = 0, max_diff_pto = 0;
    for (int i = 0; i < n * 3; i++) {
        float d1 = fabsf(f_scalar[i] - f_sve[i]);
        float d2 = fabsf(f_scalar[i] - f_pto[i]);
        if (d1 > max_diff_sve) max_diff_sve = d1;
        if (d2 > max_diff_pto) max_diff_pto = d2;
    }
    
    /* Count contiguous runs */
    int total_runs = 0, contig_runs = 0;
    for (int i = 0; i < n; i++) {
        for (int ti = 0; ti < ctx->num_tiles; ti++) {
            int rc = ctx->tile_run_count[i][ti];
            if (rc == 0) continue;
            total_runs++;
            int rs = ctx->tile_run_start[i][ti];
            int is_c = 1;
            for (int m = 1; m < rc; m++) {
                if (ctx->sorted_jatoms[nl->start[i]+rs+m] != ctx->sorted_jatoms[nl->start[i]+rs+m-1]+1) {
                    is_c = 0; break;
                }
            }
            if (is_c) contig_runs++;
        }
    }
    
    /* ---- 报告 ---- */
    printf("\n================================================================\n");
    printf("  性能报告\n");
    printf("================================================================\n");
    printf("组1 Scalar:    %.3f ms/step (1.00x)  — 基线\n",
           t_scalar / nsteps * 1000);
    printf("组2 SVE naive: %.3f ms/step (%.2fx)  — SVE向量化收益\n",
           t_sve / nsteps * 1000, t_scalar / t_sve);
    printf("组3 PTO-SVE:   %.3f ms/step (%.2fx)  — SVE+PTO融合总收益\n",
           t_pto / nsteps * 1000, t_scalar / t_pto);
    printf("\n--- 收益分解 ---\n");
    printf("SVE 向量化本身: %.2fx (组2/组1)\n", t_scalar / t_sve);
    printf("PTO 融合额外:   %.2fx (组3/组2)  ← 这才是 PTO 的贡献!\n", t_sve / t_pto);
    printf("总加速比:        %.2fx (组3/组1)\n", t_scalar / t_pto);
    printf("\n--- 验证 ---\n");
    printf("SVE vs Scalar 力差异: %.6e\n", max_diff_sve);
    printf("PTO vs Scalar 力差异: %.6e\n", max_diff_pto);
    printf("连续加载路径: %d/%d (%.1f%%)\n", contig_runs, total_runs,
           total_runs > 0 ? 100.0 * contig_runs / total_runs : 0);
    printf("================================================================\n");
    
    pto_destroy(ctx);
    free(f_scalar); free(f_sve); free(f_pto);
    free_neighbor_list(nl);
    free_gro(gro);
    
    return 0;
}
