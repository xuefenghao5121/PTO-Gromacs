/*
 * PTO-GROMACS v5 - SVE 最终优化版
 * 
 * 结合v3(tile+动态调度)和v4(排序连续加载)的优点:
 * - 排序j列表 -> 最大化连续加载 (v4)
 * - tile分组 -> 减少力合并开销 (v3)
 * - 动态调度 -> 负载均衡 (v3)
 * - inv_box -> 减少除法 (v4)
 * - 连续内存力缓冲 (v4)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <arm_sve.h>

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

typedef struct { int *start; int *jatoms; int *count; int total_pairs; } NBList;

static NBList* build_nblist(const float *x, int n, const float box[3], float cut) {
    NBList *nl = calloc(1,sizeof(NBList));
    nl->start = malloc(n*sizeof(int));
    nl->count = calloc(n,sizeof(int));
    float csq = cut*cut;
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
    nl->start[0]=0;
    for(int i=1;i<n;i++) nl->start[i]=nl->start[i-1]+nl->count[i-1];
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
    /* Sort j lists for contiguity */
    for(int i=0;i<n;i++){
        int s=nl->start[i], c=nl->count[i];
        for(int k=1;k<c;k++){
            int key=nl->jatoms[s+k];
            int m=k-1;
            while(m>=0 && nl->jatoms[s+m]>key){
                nl->jatoms[s+m+1]=nl->jatoms[s+m];
                m--;
            }
            nl->jatoms[s+m+1]=key;
        }
    }
    return nl;
}

typedef struct {
    int n;
    float *sx, *sy, *sz;
    float inv_box[3];
    int num_threads;
    float *all_lfx;
    float *all_lfy;
    float *all_lfz;
} PTOv5Ctx;

static void ptov5_repack_coords(PTOv5Ctx *ctx, const float *aos_coords) {
    int n = ctx->n;
    for (int i = 0; i < n; i++) {
        ctx->sx[i] = aos_coords[i * 3 + 0];
        ctx->sy[i] = aos_coords[i * 3 + 1];
        ctx->sz[i] = aos_coords[i * 3 + 2];
    }
}

static PTOv5Ctx* ptov5_init(const float *aos_coords, int n, const float box[3]) {
    PTOv5Ctx *ctx = calloc(1, sizeof(PTOv5Ctx));
    ctx->n = n;
    ctx->sx = malloc(n * sizeof(float));
    ctx->sy = malloc(n * sizeof(float));
    ctx->sz = malloc(n * sizeof(float));
    ctx->inv_box[0] = 1.0f / box[0];
    ctx->inv_box[1] = 1.0f / box[1];
    ctx->inv_box[2] = 1.0f / box[2];
    ctx->num_threads = omp_get_max_threads();
    ctx->all_lfx = calloc((size_t)ctx->num_threads * n, sizeof(float));
    ctx->all_lfy = calloc((size_t)ctx->num_threads * n, sizeof(float));
    ctx->all_lfz = calloc((size_t)ctx->num_threads * n, sizeof(float));
    ptov5_repack_coords(ctx, aos_coords);
    return ctx;
}

static void ptov5_destroy(PTOv5Ctx *ctx) {
    free(ctx->sx); free(ctx->sy); free(ctx->sz);
    free(ctx->all_lfx); free(ctx->all_lfy); free(ctx->all_lfz);
    free(ctx);
}

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

/* v5: Sorted j-list + contiguous load + dynamic schedule + inv_box */
static double ptov5_compute(PTOv5Ctx *ctx, float *f_aos, int n, float box[3],
                             NBList *nl, float cut) {
    float csq = cut*cut, eps = 0.5f, ssq = 0.09f;
    float *sx = ctx->sx, *sy = ctx->sy, *sz = ctx->sz;
    float inv_bx = ctx->inv_box[0], inv_by = ctx->inv_box[1], inv_bz = ctx->inv_box[2];
    int nt = ctx->num_threads;

    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    memset(f_aos, 0, n*3*sizeof(float));
    memset(ctx->all_lfx, 0, (size_t)nt*n*sizeof(float));
    memset(ctx->all_lfy, 0, (size_t)nt*n*sizeof(float));
    memset(ctx->all_lfz, 0, (size_t)nt*n*sizeof(float));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *lfx = ctx->all_lfx + (size_t)tid * n;
        float *lfy = ctx->all_lfy + (size_t)tid * n;
        float *lfz = ctx->all_lfz + (size_t)tid * n;

        svbool_t all_p = svptrue_b32();
        int vl = (int)svcntw();

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < n; i++) {
            float xi = sx[i], yi = sy[i], zi = sz[i];
            float fix = 0, fiy = 0, fiz = 0;
            int ni = nl->count[i];
            int base = nl->start[i];

            /* Walk sorted j-list, process contiguous runs */
            int k = 0;
            while (k < ni) {
                int j_start = nl->jatoms[base + k];
                int run_len = 1;
                while (k + run_len < ni &&
                       nl->jatoms[base + k + run_len] == j_start + run_len) {
                    run_len++;
                }

                for (int r = 0; r < run_len; r += vl) {
                    svbool_t pg = svwhilelt_b32(r, run_len);
                    int j0 = j_start + r;

                    svfloat32_t xj_v = svld1_f32(pg, &sx[j0]);
                    svfloat32_t yj_v = svld1_f32(pg, &sy[j0]);
                    svfloat32_t zj_v = svld1_f32(pg, &sz[j0]);

                    svfloat32_t dx_v = svsub_f32_x(pg, svdup_f32(xi), xj_v);
                    svfloat32_t dy_v = svsub_f32_x(pg, svdup_f32(yi), yj_v);
                    svfloat32_t dz_v = svsub_f32_x(pg, svdup_f32(zi), zj_v);

                    dx_v = svsub_f32_x(pg, dx_v, svmul_f32_x(pg, svdup_f32(box[0]),
                        svrinta_f32_x(pg, svmul_f32_x(pg, dx_v, svdup_f32(inv_bx)))));
                    dy_v = svsub_f32_x(pg, dy_v, svmul_f32_x(pg, svdup_f32(box[1]),
                        svrinta_f32_x(pg, svmul_f32_x(pg, dy_v, svdup_f32(inv_by)))));
                    dz_v = svsub_f32_x(pg, dz_v, svmul_f32_x(pg, svdup_f32(box[2]),
                        svrinta_f32_x(pg, svmul_f32_x(pg, dz_v, svdup_f32(inv_bz)))));

                    svfloat32_t rsq = svadd_f32_x(pg, svadd_f32_x(pg,
                        svmul_f32_x(pg, dx_v, dx_v), svmul_f32_x(pg, dy_v, dy_v)),
                        svmul_f32_x(pg, dz_v, dz_v));

                    svbool_t valid = svand_b_z(pg,
                        svcmplt_f32(pg, rsq, svdup_f32(csq)),
                        svcmpgt_f32(pg, rsq, svdup_f32(1e-8f)));

                    if (svptest_any(all_p, valid)) {
                        svfloat32_t ir = svdiv_f32_z(valid, svdup_f32(1.0f), rsq);
                        svfloat32_t s2 = svmul_f32_z(valid, svdup_f32(ssq), ir);
                        svfloat32_t s6 = svmul_f32_z(valid, svmul_f32_z(valid, s2, s2), s2);
                        svfloat32_t s12 = svmul_f32_z(valid, s6, s6);
                        svfloat32_t fr = svmul_f32_z(valid, svdup_f32(24.0f*eps),
                            svmul_f32_z(valid, svsub_f32_z(valid,
                                svmul_f32_z(valid, svdup_f32(2.0f), s12), s6), ir));

                        svfloat32_t fx_v = svmul_f32_z(valid, fr, dx_v);
                        svfloat32_t fy_v = svmul_f32_z(valid, fr, dy_v);
                        svfloat32_t fz_v = svmul_f32_z(valid, fr, dz_v);

                        fix += svadda_f32(valid, 0.0f, fx_v);
                        fiy += svadda_f32(valid, 0.0f, fy_v);
                        fiz += svadda_f32(valid, 0.0f, fz_v);

                        float fxo[16], fyo[16], fzo[16];
                        svst1_f32(pg, fxo, fx_v);
                        svst1_f32(pg, fyo, fy_v);
                        svst1_f32(pg, fzo, fz_v);
                        int rem = run_len - r;
                        int cnt = (rem < vl) ? rem : vl;
                        for (int m = 0; m < cnt; m++) {
                            int j = j0 + m;
                            lfx[j] -= fxo[m];
                            lfy[j] -= fyo[m];
                            lfz[j] -= fzo[m];
                        }
                    }
                }
                k += run_len;
            }
            lfx[i] += fix;
            lfy[i] += fiy;
            lfz[i] += fiz;
        }
    }

    /* Merge thread forces */
    for (int t = 1; t < nt; t++) {
        float *lfx = ctx->all_lfx + (size_t)t * n;
        float *lfy = ctx->all_lfy + (size_t)t * n;
        float *lfz = ctx->all_lfz + (size_t)t * n;
        #pragma omp simd
        for (int k = 0; k < n; k++) {
            ctx->all_lfx[k] += lfx[k];
            ctx->all_lfy[k] += lfy[k];
            ctx->all_lfz[k] += lfz[k];
        }
    }
    #pragma omp simd
    for (int k = 0; k < n; k++) {
        f_aos[k*3+0] = ctx->all_lfx[k];
        f_aos[k*3+1] = ctx->all_lfy[k];
        f_aos[k*3+2] = ctx->all_lfz[k];
    }

    struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
}

int main(int argc, char *argv[]) {
    if(argc<2){ printf("Usage: %s <file.gro> [cutoff] [nsteps]\n",argv[0]); return 1; }
    float cut = argc>2?atof(argv[2]):1.0f;
    int steps = argc>3?atoi(argv[3]):200;

    GroData *g = read_gro(argv[1]);
    if(!g) { fprintf(stderr,"Cannot read %s\n",argv[1]); return 1; }
    int n = g->natoms;

    printf("================================================================\n");
    printf("  PTO-GROMACS v5 - ARM SVE (Final Optimized)\n");
    printf("================================================================\n");
    printf("Atoms: %d | Box: %.1fx%.1fx%.1f | Cut: %.2f | Steps: %d\n",
           n, g->box[0],g->box[1],g->box[2], cut, steps);
    printf("SVE: %ld-bit (%ld floats)\n", svcntb()*8, svcntw());
    printf("Threads: %d\n\n", omp_get_max_threads());

    printf("Building neighbor list (sorted)..."); fflush(stdout);
    NBList *nl = build_nblist(g->x, n, g->box, cut);
    printf(" %d pairs (%.1f nbs/atom)\n", nl->total_pairs, 2.0f*nl->total_pairs/n);

    long total_j = 0, contig_j = 0;
    for(int i=0;i<n;i++){
        int s=nl->start[i], c=nl->count[i];
        total_j += c;
        for(int k=1;k<c;k++){
            if(nl->jatoms[s+k]==nl->jatoms[s+k-1]+1) contig_j++;
        }
    }
    printf("Contiguity: %ld/%ld (%.1f%%)\n\n", contig_j, total_j>0?total_j:1,
           100.0*contig_j/(total_j>0?total_j:1));

    printf("Initializing PTO v5 context..."); fflush(stdout);
    PTOv5Ctx *ctx = ptov5_init(g->x, n, g->box);
    printf(" done\n\n");

    float *f_scalar = (float*)malloc(n*3*sizeof(float));
    float *f_v5 = (float*)malloc(n*3*sizeof(float));

    printf("Scalar:      "); fflush(stdout);
    scalar_nb(g->x, f_scalar, n, g->box, nl, cut);
    double ts=0;
    for(int s=0;s<steps;s++) ts+=scalar_nb(g->x, f_scalar, n, g->box, nl, cut);
    printf("%.3f ms/step\n", ts/steps*1000);

    printf("PTO SVE v5:  "); fflush(stdout);
    ptov5_compute(ctx, f_v5, n, g->box, nl, cut);
    double tv5=0;
    for(int s=0;s<steps;s++) tv5+=ptov5_compute(ctx, f_v5, n, g->box, nl, cut);
    printf("%.3f ms/step (%.2fx)\n", tv5/steps*1000, ts/tv5);

    float max_diff = 0, rel_diff_sum = 0;
    int diff_count = 0;
    for(int k=0;k<n*3;k++){
        float d=fabsf(f_scalar[k]-f_v5[k]);
        if(d>max_diff) max_diff=d;
        if(fabsf(f_scalar[k])>1e-6f) {
            rel_diff_sum += d/fabsf(f_scalar[k]);
            diff_count++;
        }
    }

    printf("\n================================================================\n");
    printf("  Performance Report\n");
    printf("================================================================\n");
    printf("Scalar:    %.3f ms/step (1.00x)\n", ts/steps*1000);
    printf("PTO SVE:   %.3f ms/step (%.2fx)\n", tv5/steps*1000, ts/tv5);
    printf("Force max diff: %.6e\n", max_diff);
    printf("Force avg rel diff: %.6e (%d components)\n",
           diff_count>0 ? rel_diff_sum/diff_count : 0, diff_count);
    printf("================================================================\n");

    ptov5_destroy(ctx);
    free(f_scalar); free(f_v5);
    free(nl->start); free(nl->jatoms); free(nl->count); free(nl);
    free(g->x); free(g);
    return 0;
}
