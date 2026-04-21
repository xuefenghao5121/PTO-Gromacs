/*
 * PTO-GROMACS v3 - 最终重构版
 * 
 * 关键设计: 初始化一次，计算N步
 * - 预分配SoA内存，预构建tile-sorted邻居列表
 * - 每步只需: pack坐标 -> 计算 -> unpack力
 * - 消除gather/scatter: SoA布局 + tile内连续加载
 * 
 * 编译 (ARM SVE):
 *   gcc -O3 -march=armv8-a+sve -msve-vector-bits=256 -ffast-math -fopenmp \
 *       pto_e2e_v3.c -o pto_e2e_v3 -lm
 * 编译 (x86):
 *   gcc -O3 -march=native -ffast-math -fopenmp pto_e2e_v3.c -o pto_e2e_v3 -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#define HAS_SVE 1
#else
#define HAS_SVE 0
#endif

#ifdef __AVX__
#include <immintrin.h>
#define HAS_AVX 1
#else
#define HAS_AVX 0
#endif

/* ============ GRO parser ============ */
typedef struct { int natoms; float *x; float box[3]; } GroData;
static GroData* read_gro(const char *fn) {
    FILE *fp = fopen(fn,"r"); if(!fp) return NULL;
    GroData *g = calloc(1,sizeof(GroData)); char line[256];
    (void)fgets(line,sizeof(line),fp); (void)fgets(line,sizeof(line),fp);
    g->natoms = atoi(line);
    g->x = (float*)malloc(g->natoms*3*sizeof(float));
    for(int i=0;i<g->natoms;i++){
        (void)fgets(line,sizeof(line),fp);
        char *p = line+20;
        g->x[i*3+0]=strtof(p,&p); g->x[i*3+1]=strtof(p,&p); g->x[i*3+2]=strtof(p,&p);
    }
    (void)fgets(line,sizeof(line),fp);
    sscanf(line,"%f%f%f",&g->box[0],&g->box[1],&g->box[2]);
    fclose(fp); return g;
}

/* ============ Neighbor list ============ */
typedef struct { int *start; int *jatoms; int *count; int total_pairs; } NBList;
static NBList* build_nblist(const float *x, int n, const float box[3], float cut) {
    NBList *nl = (NBList*)calloc(1,sizeof(NBList));
    nl->start = (int*)malloc(n*sizeof(int));
    nl->count = (int*)calloc(n,sizeof(int));
    float csq = cut*cut; int total=0;
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++){
        float dx=x[i*3]-x[j*3], dy=x[i*3+1]-x[j*3+1], dz=x[i*3+2]-x[j*3+2];
        dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
        if(dx*dx+dy*dy+dz*dz < csq) { nl->count[i]++; total++; }
    }
    nl->start[0]=0;
    for(int i=1;i<n;i++) nl->start[i]=nl->start[i-1]+nl->count[i-1];
    nl->jatoms = (int*)malloc(total*sizeof(int));
    nl->total_pairs = total;
    int *pos = (int*)calloc(n,sizeof(int));
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++){
        float dx=x[i*3]-x[j*3], dy=x[i*3+1]-x[j*3+1], dz=x[i*3+2]-x[j*3+2];
        dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
        if(dx*dx+dy*dy+dz*dz < csq) nl->jatoms[nl->start[i]+pos[i]++]=j;
    }
    free(pos); return nl;
}

/* ============ PTO v3 Tile Context ============ */
/*
 * 预分配的Tile上下文 - 初始化一次，计算多次
 */
typedef struct {
    int n;
    int tile_size;
    int num_tiles;
    
    /* SoA coordinate arrays (pre-allocated, repacked each step) */
    float *sx, *sy, *sz;
    
    /* Thread-local force buffers (pre-allocated) */
    int nthreads;
    float **thread_lfx, **thread_lfy, **thread_lfz;
    
    /* Tile-sorted neighbor list (pre-built, reused across steps) */
    int *sorted_jatoms;          /* j atoms sorted by tile for each i */
    int **tile_run_start;        /* [n][num_tiles+1] */
    int **tile_run_count;        /* [n][num_tiles] */
    
    /* Contiguous j-index runs for SVE loading */
    /* For each (atom_i, tile_t), we store the run of j indices */
} PTOv3Ctx;

static PTOv3Ctx* ptov3_init(const float *aos_coords, int n, const NBList *nl,
                             int tile_size) {
    PTOv3Ctx *ctx = (PTOv3Ctx*)calloc(1, sizeof(PTOv3Ctx));
    ctx->n = n;
    ctx->tile_size = tile_size;
    ctx->num_tiles = (n + tile_size - 1) / tile_size;
    
    /* Pre-allocate SoA arrays */
    ctx->sx = (float*)aligned_alloc(64, n*sizeof(float));
    ctx->sy = (float*)aligned_alloc(64, n*sizeof(float));
    ctx->sz = (float*)aligned_alloc(64, n*sizeof(float));
    
    /* Pack initial coords */
    for (int i = 0; i < n; i++) {
        ctx->sx[i] = aos_coords[i*3+0];
        ctx->sy[i] = aos_coords[i*3+1];
        ctx->sz[i] = aos_coords[i*3+2];
    }
    
    /* Pre-allocate thread-local force buffers */
    ctx->nthreads = omp_get_max_threads();
    ctx->thread_lfx = (float**)malloc(ctx->nthreads * sizeof(float*));
    ctx->thread_lfy = (float**)malloc(ctx->nthreads * sizeof(float*));
    ctx->thread_lfz = (float**)malloc(ctx->nthreads * sizeof(float*));
    for (int t = 0; t < ctx->nthreads; t++) {
        ctx->thread_lfx[t] = (float*)calloc(n, sizeof(float));
        ctx->thread_lfy[t] = (float*)calloc(n, sizeof(float));
        ctx->thread_lfz[t] = (float*)calloc(n, sizeof(float));
    }
    
    /* Build tile-sorted neighbor list (one-time cost) */
    int num_tiles = ctx->num_tiles;
    ctx->sorted_jatoms = (int*)malloc(nl->total_pairs * sizeof(int));
    ctx->tile_run_start = (int**)malloc(n * sizeof(int*));
    ctx->tile_run_count = (int**)malloc(n * sizeof(int*));
    
    for (int i = 0; i < n; i++) {
        ctx->tile_run_start[i] = (int*)calloc(num_tiles + 1, sizeof(int));
        ctx->tile_run_count[i] = (int*)calloc(num_tiles, sizeof(int));
    }
    
    /* Sort neighbors by tile */
    int *tmp = (int*)malloc(nl->total_pairs * sizeof(int));
    memcpy(tmp, nl->jatoms, nl->total_pairs * sizeof(int));
    
    for (int i = 0; i < n; i++) {
        int cnt = nl->count[i];
        int base = nl->start[i];
        
        for (int k = 0; k < cnt; k++) {
            int j = tmp[base + k];
            int ti = j / tile_size;
            if (ti >= num_tiles) ti = num_tiles - 1;
            ctx->tile_run_count[i][ti]++;
        }
        
        ctx->tile_run_start[i][0] = 0;
        for (int ti = 1; ti <= num_tiles; ti++) {
            ctx->tile_run_start[i][ti] = ctx->tile_run_start[i][ti-1] + ctx->tile_run_count[i][ti-1];
        }
        
        int *pos = (int*)calloc(num_tiles, sizeof(int));
        for (int k = 0; k < cnt; k++) {
            int j = tmp[base + k];
            int ti = j / tile_size;
            if (ti >= num_tiles) ti = num_tiles - 1;
            int idx = ctx->tile_run_start[i][ti] + pos[ti]++;
            ctx->sorted_jatoms[base + idx] = j;
        }
        free(pos);
    }
    free(tmp);
    
    return ctx;
}

static void ptov3_repack_coords(PTOv3Ctx *ctx, const float *aos_coords) {
    for (int i = 0; i < ctx->n; i++) {
        ctx->sx[i] = aos_coords[i*3+0];
        ctx->sy[i] = aos_coords[i*3+1];
        ctx->sz[i] = aos_coords[i*3+2];
    }
}

static void ptov3_destroy(PTOv3Ctx *ctx) {
    if (!ctx) return;
    free(ctx->sx); free(ctx->sy); free(ctx->sz);
    for (int t = 0; t < ctx->nthreads; t++) {
        free(ctx->thread_lfx[t]);
        free(ctx->thread_lfy[t]);
        free(ctx->thread_lfz[t]);
    }
    free(ctx->thread_lfx); free(ctx->thread_lfy); free(ctx->thread_lfz);
    free(ctx->sorted_jatoms);
    for (int i = 0; i < ctx->n; i++) {
        free(ctx->tile_run_start[i]);
        free(ctx->tile_run_count[i]);
    }
    free(ctx->tile_run_start);
    free(ctx->tile_run_count);
    free(ctx);
}

/* ============ Scalar baseline ============ */
static double scalar_nb(float *x, float *f, int n, float box[3], NBList *nl, float cut) {
    float csq=cut*cut, eps=0.5f, ssq=0.09f;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
    memset(f,0,n*3*sizeof(float));
    #pragma omp parallel
    {
        float *lf = (float*)calloc(n*3,sizeof(float));
        #pragma omp for schedule(dynamic,64)
        for(int i=0;i<n;i++){
            float fx=0,fy=0,fz=0, xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
            for(int k=0;k<nl->count[i];k++){
                int j=nl->jatoms[nl->start[i]+k];
                float dx=xi-x[j*3], dy=yi-x[j*3+1], dz=zi-x[j*3+2];
                dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
                float rsq=dx*dx+dy*dy+dz*dz;
                if(rsq<csq && rsq>1e-8f){
                    float ir=1.0f/rsq, s2=ssq*ir, s6=s2*s2*s2, s12=s6*s6;
                    float fr=24.0f*eps*(2.0f*s12-s6)*ir;
                    fx+=fr*dx; fy+=fr*dy; fz+=fr*dz;
                    lf[j*3]-=fr*dx; lf[j*3+1]-=fr*dy; lf[j*3+2]-=fr*dz;
                }
            }
            lf[i*3]+=fx; lf[i*3+1]+=fy; lf[i*3+2]+=fz;
        }
        #pragma omp critical
        for(int k=0;k<n*3;k++) f[k]+=lf[k];
        free(lf);
    }
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC,&t1);
    return (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
}

/* ====================================================================
 * PTO v3 compute kernel - 消除gather/scatter的核心
 * 
 * 每个原子的邻居按tile分组，同tile内j索引连续
 * SoA布局下 sx[j], sy[j], sz[j] 连续存储
 * → SVE可连续加载，无需标量gather
 * ==================================================================== */

#if HAS_SVE
static double ptov3_compute(PTOv3Ctx *ctx, float *f_aos, int n, float box[3],
                             NBList *nl, float cut) {
    float csq=cut*cut, eps=0.5f, ssq=0.09f;
    int num_tiles = ctx->num_tiles;
    int tile_size = ctx->tile_size;
    float *sx=ctx->sx, *sy=ctx->sy, *sz=ctx->sz;
    
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
    memset(f_aos, 0, n*3*sizeof(float));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *lfx = ctx->thread_lfx[tid];
        float *lfy = ctx->thread_lfy[tid];
        float *lfz = ctx->thread_lfz[tid];
        memset(lfx, 0, n*sizeof(float));
        memset(lfy, 0, n*sizeof(float));
        memset(lfz, 0, n*sizeof(float));
        
        svbool_t all_p = svptrue_b32();
        int vl = (int)svcntw();
        
        #pragma omp for schedule(dynamic,64)
        for (int i = 0; i < n; i++) {
            float xi=sx[i], yi=sy[i], zi=sz[i];
            svfloat32_t fx_a=svdup_f32(0), fy_a=svdup_f32(0), fz_a=svdup_f32(0);
            
            /* 遍历每个tile run */
            for (int ti = 0; ti < num_tiles; ti++) {
                int run_start = ctx->tile_run_start[i][ti];
                int run_count = ctx->tile_run_count[i][ti];
                if (run_count == 0) continue;
                
                int base_j = ctx->sorted_jatoms[nl->start[i] + run_start];
                
                /* 检查是否连续 (同tile内原子索引应该连续) */
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
                        /* ★ 连续加载路径 - 无gather! */
                        int j0 = ctx->sorted_jatoms[nl->start[i] + run_start + k];
                        svfloat32_t xj = svld1_f32(pg, &sx[j0]);
                        svfloat32_t yj = svld1_f32(pg, &sy[j0]);
                        svfloat32_t zj = svld1_f32(pg, &sz[j0]);
                        dx_v = svsub_f32_x(pg, svdup_f32(xi), xj);
                        dy_v = svsub_f32_x(pg, svdup_f32(yi), yj);
                        dz_v = svsub_f32_x(pg, svdup_f32(zi), zj);
                    } else {
                        /* Fallback: gather */
                        float jx[8],jy[8],jz[8];
                        for (int m=0;m<rem;m++){
                            int j=ctx->sorted_jatoms[nl->start[i]+run_start+k+m];
                            jx[m]=xi-sx[j]; jy[m]=yi-sy[j]; jz[m]=zi-sz[j];
                        }
                        dx_v=svld1_f32(pg,jx); dy_v=svld1_f32(pg,jy); dz_v=svld1_f32(pg,jz);
                    }
                    
                    /* PBC */
                    dx_v=svsub_f32_x(pg,dx_v,svmul_f32_x(pg,svdup_f32(box[0]),
                        svrinta_f32_x(pg,svdiv_f32_x(pg,dx_v,svdup_f32(box[0])))));
                    dy_v=svsub_f32_x(pg,dy_v,svmul_f32_x(pg,svdup_f32(box[1]),
                        svrinta_f32_x(pg,svdiv_f32_x(pg,dy_v,svdup_f32(box[1])))));
                    dz_v=svsub_f32_x(pg,dz_v,svmul_f32_x(pg,svdup_f32(box[2]),
                        svrinta_f32_x(pg,svdiv_f32_x(pg,dz_v,svdup_f32(box[2])))));
                    
                    svfloat32_t rsq = svadd_f32_x(pg, svadd_f32_x(pg,
                        svmul_f32_x(pg,dx_v,dx_v), svmul_f32_x(pg,dy_v,dy_v)),
                        svmul_f32_x(pg,dz_v,dz_v));
                    
                    svbool_t valid = svand_b_z(pg,
                        svcmplt_f32(pg,rsq,svdup_f32(csq)),
                        svcmpgt_f32(pg,rsq,svdup_f32(1e-8f)));
                    
                    if(svptest_any(all_p,valid)){
                        svfloat32_t ir=svdiv_f32_z(valid,svdup_f32(1.0f),rsq);
                        svfloat32_t s2=svmul_f32_z(valid,svdup_f32(ssq),ir);
                        svfloat32_t s4=svmul_f32_z(valid,s2,s2);
                        svfloat32_t s6=svmul_f32_z(valid,s4,s2);
                        svfloat32_t s12=svmul_f32_z(valid,s6,s6);
                        svfloat32_t fr=svmul_f32_z(valid,svdup_f32(24.0f*eps),
                            svmul_f32_z(valid,svsub_f32_z(valid,svmul_f32_z(valid,svdup_f32(2.0f),s12),s6),ir));
                        
                        svfloat32_t fx_v=svmul_f32_z(valid,fr,dx_v);
                        svfloat32_t fy_v=svmul_f32_z(valid,fr,dy_v);
                        svfloat32_t fz_v=svmul_f32_z(valid,fr,dz_v);
                        
                        fx_a=svadd_f32_m(valid,fx_a,fx_v);
                        fy_a=svadd_f32_m(valid,fy_a,fy_v);
                        fz_a=svadd_f32_m(valid,fz_a,fz_v);
                        
                        /* Accumulate j forces */
                        float fxo[8],fyo[8],fzo[8];
                        svst1_f32(pg,fxo,fx_v); svst1_f32(pg,fyo,fy_v); svst1_f32(pg,fzo,fz_v);
                        for(int m=0;m<rem;m++){
                            int j=ctx->sorted_jatoms[nl->start[i]+run_start+k+m];
                            lfx[j]-=fxo[m]; lfy[j]-=fyo[m]; lfz[j]-=fzo[m];
                        }
                    }
                }
            }
            lfx[i]+=svaddv_f32(all_p,fx_a);
            lfy[i]+=svaddv_f32(all_p,fy_a);
            lfz[i]+=svaddv_f32(all_p,fz_a);
        }
        
        #pragma omp critical
        for(int k=0;k<n;k++){
            f_aos[k*3+0]+=lfx[k];
            f_aos[k*3+1]+=lfy[k];
            f_aos[k*3+2]+=lfz[k];
        }
    }
    
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC,&t1);
    return (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
}
#endif

/* x86 scalar tile version */
static double ptov3_compute_scalar(PTOv3Ctx *ctx, float *f_aos, int n, float box[3],
                                    NBList *nl, float cut) {
    float csq=cut*cut, eps=0.5f, ssq=0.09f;
    int num_tiles = ctx->num_tiles;
    float *sx=ctx->sx, *sy=ctx->sy, *sz=ctx->sz;
    
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
    memset(f_aos, 0, n*3*sizeof(float));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *lfx = ctx->thread_lfx[tid];
        float *lfy = ctx->thread_lfy[tid];
        float *lfz = ctx->thread_lfz[tid];
        memset(lfx, 0, n*sizeof(float));
        memset(lfy, 0, n*sizeof(float));
        memset(lfz, 0, n*sizeof(float));
        
        #pragma omp for schedule(dynamic,64)
        for (int i = 0; i < n; i++) {
            float xi=sx[i], yi=sy[i], zi=sz[i];
            float fx=0, fy=0, fz=0;
            
            for (int ti = 0; ti < num_tiles; ti++) {
                int run_start = ctx->tile_run_start[i][ti];
                int run_count = ctx->tile_run_count[i][ti];
                if (run_count == 0) continue;
                
                for (int k = 0; k < run_count; k++) {
                    int j = ctx->sorted_jatoms[nl->start[i] + run_start + k];
                    float dx = xi - sx[j];
                    float dy = yi - sy[j];
                    float dz = zi - sz[j];
                    dx -= box[0]*roundf(dx/box[0]);
                    dy -= box[1]*roundf(dy/box[1]);
                    dz -= box[2]*roundf(dz/box[2]);
                    float rsq = dx*dx + dy*dy + dz*dz;
                    if (rsq < csq && rsq > 1e-8f) {
                        float ir = 1.0f/rsq, s2=ssq*ir, s6=s2*s2*s2, s12=s6*s6;
                        float fr = 24.0f*eps*(2.0f*s12-s6)*ir;
                        fx+=fr*dx; fy+=fr*dy; fz+=fr*dz;
                        lfx[j]-=fr*dx; lfy[j]-=fr*dy; lfz[j]-=fr*dz;
                    }
                }
            }
            lfx[i]+=fx; lfy[i]+=fy; lfz[i]+=fz;
        }
        
        #pragma omp critical
        for(int k=0;k<n;k++){
            f_aos[k*3+0]+=lfx[k];
            f_aos[k*3+1]+=lfy[k];
            f_aos[k*3+2]+=lfz[k];
        }
    }
    
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC,&t1);
    return (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
}

/* ====================================================================
 * PTO v3 compute kernel - x86 AVX 向量化路径
 * 
 * 修复问题E: 添加完整的 AVX 向量化路径
 * 参考 ARM SVE 版本的融合模式:
 * - 累加器保持在 YMM 寄存器
 * - 向量级别计算距离平方和力
 * - SoA + tile 分组实现连续加载
 * ==================================================================== */
#if HAS_AVX && !HAS_SVE
static double ptov3_compute_avx(PTOv3Ctx *ctx, float *f_aos, int n, float box[3],
                                NBList *nl, float cut) {
    float csq = cut*cut, eps = 0.5f, ssq = 0.09f;
    int num_tiles = ctx->num_tiles;
    float *sx = ctx->sx, *sy = ctx->sy, *sz = ctx->sz;
    
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    memset(f_aos, 0, n*3*sizeof(float));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *lfx = ctx->thread_lfx[tid];
        float *lfy = ctx->thread_lfy[tid];
        float *lfz = ctx->thread_lfz[tid];
        memset(lfx, 0, n*sizeof(float));
        memset(lfy, 0, n*sizeof(float));
        memset(lfz, 0, n*sizeof(float));
        
        /* AVX 常量 */
        __m256 vone = _mm256_set1_ps(1.0f);
        __m256 vzero = _mm256_setzero_ps();
        
        #pragma omp for schedule(dynamic,64)
        for (int i = 0; i < n; i++) {
            float xi = sx[i], yi = sy[i], zi = sz[i];
            
            /* ★ 向量累加器 - 参考 SVE 的 fx_a/fy_a/fz_a */
            __m256 vfx_a = _mm256_setzero_ps();
            __m256 vfy_a = _mm256_setzero_ps();
            __m256 vfz_a = _mm256_setzero_ps();
            
            for (int ti = 0; ti < num_tiles; ti++) {
                int run_start = ctx->tile_run_start[i][ti];
                int run_count = ctx->tile_run_count[i][ti];
                if (run_count == 0) continue;
                
                int base = nl->start[i] + run_start;
                
                /* ★ 检查整个 run 是否连续（同 SVE 版本优化） */
                int is_contig = 1;
                if (run_count >= 2) {
                    for (int m = 1; m < run_count; m++) {
                        int j_prev = ctx->sorted_jatoms[base + m - 1];
                        int j_curr = ctx->sorted_jatoms[base + m];
                        if (j_curr != j_prev + 1) { is_contig = 0; break; }
                    }
                }
                
                for (int k = 0; k < run_count; k += 8) {
                    int rem = run_count - k;
                    int cnt = (rem < 8) ? rem : 8;
                    
                    __m256 vxj, vyj, vzj;
                    int jj_buf[8];
                    
                    if (is_contig && cnt == 8) {
                        /* ★ 连续加载路径: 直接从 SoA 数组加载，无需 gather 到栈 */
                        int j0 = ctx->sorted_jatoms[base + k];
                        vxj = _mm256_loadu_ps(&sx[j0]);
                        vyj = _mm256_loadu_ps(&sy[j0]);
                        vzj = _mm256_loadu_ps(&sz[j0]);
                        for (int m = 0; m < 8; m++) jj_buf[m] = j0 + m;
                    } else {
                        /* Fallback: gather 到 SoA 缓冲区 */
                        float xj_buf[8] __attribute__((aligned(32)));
                        float yj_buf[8] __attribute__((aligned(32)));
                        float zj_buf[8] __attribute__((aligned(32)));
                        
                        for (int m = 0; m < cnt; m++) {
                            int j = ctx->sorted_jatoms[base + k + m];
                            jj_buf[m] = j;
                            xj_buf[m] = sx[j];
                            yj_buf[m] = sy[j];
                            zj_buf[m] = sz[j];
                        }
                        for (int m = cnt; m < 8; m++) {
                            jj_buf[m] = i;
                            xj_buf[m] = xi;
                            yj_buf[m] = yi;
                            zj_buf[m] = zi;
                        }
                        vxj = _mm256_load_ps(xj_buf);
                        vyj = _mm256_load_ps(yj_buf);
                        vzj = _mm256_load_ps(zj_buf);
                    }
                    
                    /* 距离向量 */
                    __m256 dx = _mm256_sub_ps(_mm256_set1_ps(xi), vxj);
                    __m256 dy = _mm256_sub_ps(_mm256_set1_ps(yi), vyj);
                    __m256 dz = _mm256_sub_ps(_mm256_set1_ps(zi), vzj);
                    
                    /* PBC (简化: box > 0) */
                    __m256 vbox_x = _mm256_set1_ps(box[0]);
                    __m256 vbox_y = _mm256_set1_ps(box[1]);
                    __m256 vbox_z = _mm256_set1_ps(box[2]);
                    
                    dx = _mm256_sub_ps(dx, _mm256_mul_ps(vbox_x,
                         _mm256_round_ps(_mm256_div_ps(dx, vbox_x), _MM_FROUND_TO_NEAREST_INT)));
                    dy = _mm256_sub_ps(dy, _mm256_mul_ps(vbox_y,
                         _mm256_round_ps(_mm256_div_ps(dy, vbox_y), _MM_FROUND_TO_NEAREST_INT)));
                    dz = _mm256_sub_ps(dz, _mm256_mul_ps(vbox_z,
                         _mm256_round_ps(_mm256_div_ps(dz, vbox_z), _MM_FROUND_TO_NEAREST_INT)));
                    
                    /* rsq */
                    __m256 rsq = _mm256_add_ps(_mm256_add_ps(
                                    _mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
                                    _mm256_mul_ps(dz, dz));
                    
                    /* mask: rsq < csq && rsq > 1e-8 */
                    __m256 mask_c = _mm256_cmp_ps(rsq, _mm256_set1_ps(csq), _CMP_LT_OS);
                    __m256 mask_n = _mm256_cmp_ps(rsq, _mm256_set1_ps(1e-8f), _CMP_GT_OS);
                    __m256 mask = _mm256_and_ps(mask_c, mask_n);
                    
                    if (_mm256_movemask_ps(mask) == 0) continue;
                    
                    /* LJ 力计算 (向量化) */
                    __m256 vrsq_safe = _mm256_max_ps(rsq, _mm256_set1_ps(1e-12f));
                    __m256 vinv_rsq = _mm256_div_ps(vone, vrsq_safe);
                    __m256 vs2 = _mm256_mul_ps(_mm256_set1_ps(ssq), vinv_rsq);
                    __m256 vs4 = _mm256_mul_ps(vs2, vs2);
                    __m256 vs6 = _mm256_mul_ps(vs4, vs2);
                    __m256 vs12 = _mm256_mul_ps(vs6, vs6);
                    
#ifdef __AVX2__
                    /* FMA: fr = 24*eps * (2*s12 - s6) * inv_rsq */
                    __m256 vterm = _mm256_fmsub_ps(_mm256_set1_ps(2.0f), vs12, vs6);
#else
                    __m256 vterm = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), vs12), vs6);
#endif
                    __m256 vfr = _mm256_mul_ps(_mm256_set1_ps(24.0f * eps),
                                 _mm256_mul_ps(vterm, vinv_rsq));
                    
                    /* 应用 mask */
                    vfr = _mm256_and_ps(vfr, mask);
                    
                    __m256 vfx = _mm256_mul_ps(vfr, dx);
                    __m256 vfy = _mm256_mul_ps(vfr, dy);
                    __m256 vfz = _mm256_mul_ps(vfr, dz);
                    
                    /* ★ 累加到向量累加器 */
                    vfx_a = _mm256_add_ps(vfx_a, vfx);
                    vfy_a = _mm256_add_ps(vfy_a, vfy);
                    vfz_a = _mm256_add_ps(vfz_a, vfz);
                    
                    /* scatter j 力 (牛顿第三定律) */
                    float fxo[8], fyo[8], fzo[8];
                    _mm256_storeu_ps(fxo, vfx);
                    _mm256_storeu_ps(fyo, vfy);
                    _mm256_storeu_ps(fzo, vfz);
                    for (int m = 0; m < cnt; m++) {
                        int j = jj_buf[m];
                        lfx[j] -= fxo[m];
                        lfy[j] -= fyo[m];
                        lfz[j] -= fzo[m];
                    }
                }
            }
            
            /* ★ 水平求和累加器，写回 i 原子力 */
            __m256 hfx = _mm256_hadd_ps(vfx_a, vfy_a);
            __m256 hfz = _mm256_hadd_ps(vfz_a, vzero);
            hfx = _mm256_hadd_ps(hfx, hfz);
            __m128 lo = _mm256_castps256_ps128(hfx);
            __m128 hi = _mm256_extractf128_ps(hfx, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            
            lfx[i] += _mm_cvtss_f32(sum);
            lfy[i] += _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1,1,1,1)));
            lfz[i] += _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2,2,2,2)));
        }
        
        #pragma omp critical
        for (int k = 0; k < n; k++) {
            f_aos[k*3+0] += lfx[k];
            f_aos[k*3+1] += lfy[k];
            f_aos[k*3+2] += lfz[k];
        }
    }
    
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
}
#endif /* HAS_AVX && !HAS_SVE */

/* ============ Main ============ */
int main(int argc, char *argv[]) {
    if(argc<2){ printf("Usage: %s <file.gro> [cutoff] [nsteps] [tile_size]\n",argv[0]); return 1; }
    float cut = argc>2?atof(argv[2]):1.0f;
    int steps = argc>3?atoi(argv[3]):200;
    int tile_size = argc>4?atoi(argv[4]):64;
    
    GroData *g = read_gro(argv[1]);
    if(!g) { fprintf(stderr,"Cannot read %s\n",argv[1]); return 1; }
    int n = g->natoms;
    
    printf("================================================================\n");
    printf("  PTO-GROMACS v3 - Tile数据重构版\n");
    printf("================================================================\n");
    printf("Atoms: %d | Box: %.1fx%.1fx%.1f | Cut: %.2f | Steps: %d | Tile: %d\n",
           n, g->box[0],g->box[1],g->box[2], cut, steps, tile_size);
#if HAS_SVE
    printf("SVE: %d-bit (%d floats)\n", svcntb()*8, svcntw());
#elif HAS_AVX
    printf("Mode: x86 AVX/AVX2 (256-bit, 8 floats)\n");
#else
    printf("Mode: x86 scalar\n");
#endif
    printf("Threads: %d\n\n", omp_get_max_threads());
    
    /* Build neighbor list */
    printf("Building neighbor list..."); fflush(stdout);
    NBList *nl = build_nblist(g->x, n, g->box, cut);
    printf(" %d pairs (%.1f nbs/atom)\n\n", nl->total_pairs, 2.0f*nl->total_pairs/n);
    
    /* Init PTO v3 context (one-time) */
    printf("Initializing PTO v3 context..."); fflush(stdout);
    PTOv3Ctx *ctx = ptov3_init(g->x, n, nl, tile_size);
    printf(" done (%d tiles)\n\n", ctx->num_tiles);
    
    float *f_scalar = (float*)malloc(n*3*sizeof(float));
    float *f_v3 = (float*)malloc(n*3*sizeof(float));
    
    /* ---- Scalar baseline ---- */
    printf("Scalar:      "); fflush(stdout);
    scalar_nb(g->x, f_scalar, n, g->box, nl, cut);
    double ts=0;
    for(int s=0;s<steps;s++) ts+=scalar_nb(g->x, f_scalar, n, g->box, nl, cut);
    printf("%.3f ms/step\n", ts/steps*1000);
    
    /* ---- PTO v3 ---- */
#if HAS_SVE
    printf("PTO v3 SVE:  "); fflush(stdout);
    ptov3_compute(ctx, f_v3, n, g->box, nl, cut);
    double tv3=0;
    for(int s=0;s<steps;s++) tv3+=ptov3_compute(ctx, f_v3, n, g->box, nl, cut);
#elif HAS_AVX
    printf("PTO v3 AVX:  "); fflush(stdout);
    ptov3_compute_avx(ctx, f_v3, n, g->box, nl, cut);
    double tv3=0;
    for(int s=0;s<steps;s++) tv3+=ptov3_compute_avx(ctx, f_v3, n, g->box, nl, cut);
#else
    printf("PTO v3 Tile: "); fflush(stdout);
    ptov3_compute_scalar(ctx, f_v3, n, g->box, nl, cut);
    double tv3=0;
    for(int s=0;s<steps;s++) tv3+=ptov3_compute_scalar(ctx, f_v3, n, g->box, nl, cut);
#endif
    printf("%.3f ms/step (%.2fx)\n", tv3/steps*1000, ts/tv3);
    
    /* Force validation */
    float max_diff = 0;
    for(int k=0;k<n*3;k++){float d=fabsf(f_scalar[k]-f_v3[k]); if(d>max_diff) max_diff=d;}
    
    /* Count contiguous runs */
    int total_runs = 0, contig_runs = 0;
    for (int i = 0; i < n; i++) {
        for (int ti = 0; ti < ctx->num_tiles; ti++) {
            int rc = ctx->tile_run_count[i][ti];
            if (rc == 0) continue;
            total_runs++;
            int rs = ctx->tile_run_start[i][ti];
            int is_c = 1;
            for (int m=1;m<rc;m++){
                if(ctx->sorted_jatoms[nl->start[i]+rs+m] != ctx->sorted_jatoms[nl->start[i]+rs+m-1]+1){
                    is_c=0; break;
                }
            }
            if(is_c) contig_runs++;
        }
    }
    
    printf("\n================================================================\n");
    printf("  性能报告\n");
    printf("================================================================\n");
    printf("Scalar:    %.3f ms/step (1.00x)\n", ts/steps*1000);
#if HAS_SVE
    printf("PTO v3:    %.3f ms/step (%.2fx)\n", tv3/steps*1000, ts/tv3);
#elif HAS_AVX
    printf("PTO v3:    %.3f ms/step (%.2fx) [x86 AVX/AVX2]\n", tv3/steps*1000, ts/tv3);
#else
    printf("PTO v3:    %.3f ms/step (%.2fx) [x86 scalar fallback]\n", tv3/steps*1000, ts/tv3);
#endif
    printf("Force max diff: %.6e\n", max_diff);
    printf("Contiguous runs: %d/%d (%.1f%%)\n", contig_runs, total_runs,
           total_runs>0 ? 100.0*contig_runs/total_runs : 0);
    printf("================================================================\n");
    
    ptov3_destroy(ctx);
    free(f_scalar); free(f_v3);
    free(nl->start); free(nl->jatoms); free(nl->count); free(nl);
    free(g->x); free(g);
    return 0;
}
