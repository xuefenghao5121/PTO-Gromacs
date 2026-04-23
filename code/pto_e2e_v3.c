/*
 * PTO-GROMACS v3 - SVE 版本
 * 
 * 简化实现，只保留 ARM SVE 路径
 * 
 * 关键设计: 初始化一次，计算N步
 * - 预分配SoA内存，预构建tile-sorted邻居列表
 * - 每步只需: pack坐标 -> 计算 -> unpack力
 * - 消除gather/scatter: SoA布局 + tile内连续加载
 * 
 * 编译 (鲲鹏930):
 *   gcc -O3 -march=armv9-a+sve+sve2 -msve-vector-bits=256 -ffast-math -fopenmp \
 *       pto_e2e_v3.c -o pto_e2e_v3 -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <arm_sve.h>

/* ============ GRO parser ============ */
typedef struct { int natoms; float *x; float box[3]; } GroData;
static GroData* read_gro(const char *fn) {
    FILE *fp = fopen(fn,"r"); if(!fp) return NULL;
    GroData *g = calloc(1,sizeof(GroData)); char line[256];
    fgets(line,sizeof(line),fp); fgets(line,sizeof(line),fp);
    g->natoms = atoi(line);
    g->x = malloc(g->natoms*3*sizeof(float));
    for(int i=0;i<g->natoms;i++){
        fgets(line,sizeof(line),fp);
        char *p = line+20;
        g->x[i*3+0]=strtof(p,&p); g->x[i*3+1]=strtof(p,&p); g->x[i*3+2]=strtof(p,&p);
    }
    fgets(line,sizeof(line),fp);
    sscanf(line,"%f%f%f",&g->box[0],&g->box[1],&g->box[2]);
    fclose(fp); return g;
}

/* ============ Neighbor list ============ */
typedef struct { int *start; int *jatoms; int *count; int total_pairs; } NBList;

static NBList* build_nblist(const float *x, int n, const float box[3], float cut) {
    NBList *nl = calloc(1,sizeof(NBList));
    nl->start = malloc(n*sizeof(int));
    nl->count = calloc(n,sizeof(int));
    float csq = cut*cut;
    
    /* Count pass */
    for(int i=0;i<n;i++){
        float xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
        int cnt=0;
        for(int j=i+1;j<n;j++){
            float dx=xi-x[j*3],dy=yi-x[j*3+1],dz=zi-x[j*3+2];
            dx-=box[0]*rintf(dx/box[0]);
            dy-=box[1]*rintf(dy/box[1]);
            dz-=box[2]*rintf(dz/box[2]);
            if(dx*dx+dy*dy+dz*dz<csq) cnt++;
        }
        nl->count[i]=cnt;
        nl->total_pairs+=cnt;
    }
    
    /* Prefix sum */
    nl->start[0]=0;
    for(int i=1;i<n;i++) nl->start[i]=nl->start[i-1]+nl->count[i-1];
    
    /* Fill pass */
    nl->jatoms = malloc(nl->total_pairs*sizeof(int));
    int *pos = calloc(n,sizeof(int));
    for(int i=0;i<n;i++){
        float xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
        for(int j=i+1;j<n;j++){
            float dx=xi-x[j*3],dy=yi-x[j*3+1],dz=zi-x[j*3+2];
            dx-=box[0]*rintf(dx/box[0]);
            dy-=box[1]*rintf(dy/box[1]);
            dz-=box[2]*rintf(dz/box[2]);
            if(dx*dx+dy*dy+dz*dz<csq)
                nl->jatoms[nl->start[i]+pos[i]++]=j;
        }
    }
    free(pos);
    return nl;
}

/* ============ PTO v3 Context ============ */
typedef struct {
    int n;
    int num_tiles;
    int tile_size;
    
    /* SoA coordinates */
    float *sx, *sy, *sz;
    
    /* Per-thread SoA force buffers */
    float **thread_lfx, **thread_lfy, **thread_lfz;
    
    /* Tile-sorted neighbor list */
    int **tile_run_start;  /* [n][num_tiles] */
    int **tile_run_count;  /* [n][num_tiles] */
    int  *sorted_jatoms;   /* sorted j indices */
} PTOv3Ctx;

static void ptov3_repack_coords(PTOv3Ctx *ctx, const float *aos_coords) {
    int n = ctx->n;
    for (int i = 0; i < n; i++) {
        ctx->sx[i] = aos_coords[i * 3 + 0];
        ctx->sy[i] = aos_coords[i * 3 + 1];
        ctx->sz[i] = aos_coords[i * 3 + 2];
    }
}

static PTOv3Ctx* ptov3_init(const float *aos_coords, int n, const NBList *nl,
                             int tile_size) {
    PTOv3Ctx *ctx = calloc(1, sizeof(PTOv3Ctx));
    ctx->n = n;
    ctx->tile_size = tile_size;
    ctx->num_tiles = (n + tile_size - 1) / tile_size;
    
    /* SoA memory */
    ctx->sx = malloc(n * sizeof(float));
    ctx->sy = malloc(n * sizeof(float));
    ctx->sz = malloc(n * sizeof(float));
    
    /* Per-thread force buffers */
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
    
    /* Sort j atoms by tile for each i */
    int *tile_tmp = malloc(ctx->num_tiles * sizeof(int));
    for (int i = 0; i < n; i++) {
        int ni = nl->count[i];
        int *js = &nl->jatoms[nl->start[i]];
        
        /* Count per tile */
        memset(tile_tmp, 0, ctx->num_tiles * sizeof(int));
        for (int k = 0; k < ni; k++) {
            int ti = js[k] / tile_size;
            if (ti >= ctx->num_tiles) ti = ctx->num_tiles - 1;
            tile_tmp[ti]++;
        }
        
        /* Compute run_start */
        int offset = 0;
        for (int ti = 0; ti < ctx->num_tiles; ti++) {
            ctx->tile_run_count[i][ti] = tile_tmp[ti];
            ctx->tile_run_start[i][ti] = offset;
            offset += tile_tmp[ti];
        }
        
        /* Place sorted j atoms */
        int *pos = calloc(ctx->num_tiles, sizeof(int));
        for (int k = 0; k < ni; k++) {
            int j = js[k];
            int ti = j / tile_size;
            if (ti >= ctx->num_tiles) ti = ctx->num_tiles - 1;
            int idx = nl->start[i] + ctx->tile_run_start[i][ti] + pos[ti];
            ctx->sorted_jatoms[idx] = j;
            pos[ti]++;
        }
        free(pos);
    }
    free(tile_tmp);
    
    /* Initial pack */
    ptov3_repack_coords(ctx, aos_coords);
    
    return ctx;
}

static void ptov3_destroy(PTOv3Ctx *ctx) {
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
    free(ctx->tile_run_start);
    free(ctx->tile_run_count);
    free(ctx->sorted_jatoms);
    free(ctx);
}

/* ============ Scalar baseline ============ */
static double scalar_nb(float *x, float *f, int n, float box[3], NBList *nl, float cut) {
    float csq=cut*cut;
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    memset(f,0,n*3*sizeof(float));
    for(int i=0;i<n;i++){
        float xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
        for(int k=0;k<nl->count[i];k++){
            int j=nl->jatoms[nl->start[i]+k];
            float dx=xi-x[j*3],dy=yi-x[j*3+1],dz=zi-x[j*3+2];
            dx-=box[0]*rintf(dx/box[0]); dy-=box[1]*rintf(dy/box[1]); dz-=box[2]*rintf(dz/box[2]);
            float rsq=dx*dx+dy*dy+dz*dz;
            if(rsq<csq && rsq>1e-8f){
                float ir=1.0f/rsq, s2=0.09f*ir, s6=s2*s2*s2, s12=s6*s6;
                float fr=24.0f*0.5f*(2.0f*s12-s6)*ir;
                float fx=fr*dx, fy=fr*dy, fz=fr*dz;
                f[i*3]+=fx; f[i*3+1]+=fy; f[i*3+2]+=fz;
                f[j*3]-=fx; f[j*3+1]-=fy; f[j*3+2]-=fz;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    return (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
}

/* ====================================================================
 * PTO v3 compute kernel - ARM SVE 向量化路径
 * 
 * 核心优化:
 * - 累加器保持在 SVE 向量寄存器
 * - SoA + tile 分组实现连续加载 (svld1)
 * - 谓词化条件执行，零开销分支
 * ==================================================================== */
static double ptov3_compute(PTOv3Ctx *ctx, float *f_aos, int n, float box[3],
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
                        /* Fallback: gather to temp then load */
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
    printf("  PTO-GROMACS v3 - ARM SVE\n");
    printf("================================================================\n");
    printf("Atoms: %d | Box: %.1fx%.1fx%.1f | Cut: %.2f | Steps: %d | Tile: %d\n",
           n, g->box[0],g->box[1],g->box[2], cut, steps, tile_size);
    printf("SVE: %ld-bit (%ld floats)\n", svcntb()*8, svcntw());
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
    
    /* ---- PTO v3 SVE ---- */
    printf("PTO v3 SVE:  "); fflush(stdout);
    ptov3_compute(ctx, f_v3, n, g->box, nl, cut);
    double tv3=0;
    for(int s=0;s<steps;s++) tv3+=ptov3_compute(ctx, f_v3, n, g->box, nl, cut);
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
    printf("PTO SVE:   %.3f ms/step (%.2fx)\n", tv3/steps*1000, ts/tv3);
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
