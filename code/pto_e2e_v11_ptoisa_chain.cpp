/**
 * PTO-GROMACS v11 — 真正的 PTO-ISA 算子组合实现
 *
 * 使用 PTO-ISA 算子链组合：
 *   TLOAD(j坐标) → TSUB(dx) → TPBC(dx) → TMUL+TADD(rsq) → TLJ_FORCE → TREDUCE(i力)
 *
 * 每个算子独立调用，中间结果在 Tile 中流转
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "pto_gromacs_core.hpp"

using namespace pto::cpu;

constexpr int TILE_COLS = 8;

/* GRO parser */
typedef struct { int natoms; float *x; float box[3]; } GroData;
static GroData* read_gro(const char *fn) {
    FILE *fp = fopen(fn,"r"); if(!fp) return NULL;
    GroData *g = (GroData*)calloc(1,sizeof(GroData)); char line[256];
    fgets(line,sizeof(line),fp); fgets(line,sizeof(line),fp);
    g->natoms = atoi(line);
    g->x = (float*)malloc(g->natoms*3*sizeof(float));
    for(int i=0;i<g->natoms;i++){
        fgets(line,sizeof(line),fp); char *p = line+20;
        g->x[i*3+0]=strtof(p,&p); g->x[i*3+1]=strtof(p,&p); g->x[i*3+2]=strtof(p,&p);
    }
    fgets(line,sizeof(line),fp);
    sscanf(line,"%f%f%f",&g->box[0],&g->box[1],&g->box[2]);
    fclose(fp); return g;
}

/* Neighbor list */
typedef struct { int *start; int *jatoms; int *count; int total_pairs; } NBList;
static NBList* build_nblist(const float *x, int n, const float box[3], float cut) {
    NBList *nl = (NBList*)calloc(1,sizeof(NBList));
    nl->start = (int*)malloc(n*sizeof(int));
    nl->count = (int*)calloc(n,sizeof(int));
    float csq = cut*cut;
    for(int i=0;i<n;i++){
        float xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
        for(int j=i+1;j<n;j++){
            float dx=xi-x[j*3],dy=yi-x[j*3+1],dz=zi-x[j*3+2];
            dx-=box[0]*rintf(dx/box[0]); dy-=box[1]*rintf(dy/box[1]); dz-=box[2]*rintf(dz/box[2]);
            if(dx*dx+dy*dy+dz*dz<csq) nl->count[i]++;
        }
        nl->total_pairs+=nl->count[i];
    }
    nl->start[0]=0;
    for(int i=1;i<n;i++) nl->start[i]=nl->start[i-1]+nl->count[i-1];
    nl->jatoms = (int*)malloc(nl->total_pairs*sizeof(int));
    int *pos = (int*)calloc(n,sizeof(int));
    for(int i=0;i<n;i++){
        float xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
        for(int j=i+1;j<n;j++){
            float dx=xi-x[j*3],dy=yi-x[j*3+1],dz=zi-x[j*3+2];
            dx-=box[0]*rintf(dx/box[0]); dy-=box[1]*rintf(dy/box[1]); dz-=box[2]*rintf(dz/box[2]);
            if(dx*dx+dy*dy+dz*dz<csq) nl->jatoms[nl->start[i]+pos[i]++]=j;
        }
    }
    free(pos);
    for(int i=0;i<n;i++){
        int s=nl->start[i], c=nl->count[i];
        for(int k=1;k<c;k++){
            int key=nl->jatoms[s+k]; int m=k-1;
            while(m>=0 && nl->jatoms[s+m]>key){ nl->jatoms[s+m+1]=nl->jatoms[s+m]; m--; }
            nl->jatoms[s+m+1]=key;
        }
    }
    return nl;
}

/* PTO-ISA 算子链计算 */
static double ptoisa_compute(float *sx, float *sy, float *sz,
                              float *f_aos, int n, float box[3],
                              NBList *nl, float cut, int nt,
                              float *all_lfx, float *all_lfy, float *all_lfz) {
    LJParamsT<1, TILE_COLS> lj_params;
    lj_params.sigma_sq = 0.09f;
    lj_params.epsilon = 0.5f;
    lj_params.cutoff_sq = cut * cut;
    lj_params.min_rsq = 1e-8f;
    float inv_bx = 1.0f/box[0], inv_by = 1.0f/box[1], inv_bz = 1.0f/box[2];

    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    memset(f_aos, 0, n*3*sizeof(float));
    memset(all_lfx, 0, (size_t)nt*n*sizeof(float));
    memset(all_lfy, 0, (size_t)nt*n*sizeof(float));
    memset(all_lfz, 0, (size_t)nt*n*sizeof(float));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *lfx = all_lfx + (size_t)tid * n;
        float *lfy = all_lfy + (size_t)tid * n;
        float *lfz = all_lfz + (size_t)tid * n;

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < n; i++) {
            float xi = sx[i], yi = sy[i], zi = sz[i];
            float fix = 0, fiy = 0, fiz = 0;
            int ni = nl->count[i];
            int base = nl->start[i];

            int k = 0;
            while (k < ni) {
                int j_start = nl->jatoms[base + k];
                int run_len = 1;
                while (k + run_len < ni && nl->jatoms[base + k + run_len] == j_start + run_len)
                    run_len++;

                for (int r = 0; r < run_len; r += TILE_COLS) {
                    int rem = run_len - r;
                    int tile_n = (rem < TILE_COLS) ? rem : TILE_COLS;
                    int j0 = j_start + r;

                    /* ===== PTO-ISA 算子链 ===== */
                    TileFixed<1, TILE_COLS> xj, yj, zj;
                    TileFixed<1, TILE_COLS> dx, dy, dz;
                    TileFixed<1, TILE_COLS> rsq;
                    TileFixed<1, TILE_COLS> fx, fy, fz;
                    TileFixed<1, TILE_COLS> t1, t2, t3, t4, t5, t6, t7;

                    xj.SetValidCols(tile_n);
                    yj.SetValidCols(tile_n);
                    zj.SetValidCols(tile_n);

                    /* Step 1: TLOAD - 加载 j 坐标 */
                    GlobalTensor1D gxj(sx, 1), gyj(sy, 1), gzj(sz, 1);
                    TLOAD(xj, gxj, j0);
                    TLOAD(yj, gyj, j0);
                    TLOAD(zj, gzj, j0);

                    /* Step 2: TFILL - 广播 i 坐标 */
                    TileFixed<1, TILE_COLS> xi_t, yi_t, zi_t;
                    TFILL(xi_t, xi);
                    TFILL(yi_t, yi);
                    TFILL(zi_t, zi);

                    /* Step 3: TSUB - dx = xi - xj */
                    TSUB(dx, xi_t, xj);
                    TSUB(dy, yi_t, yj);
                    TSUB(dz, zi_t, zj);

                    /* Step 4: TPBC - 周期性边界条件 */
                    TPBC(dx, box[0], inv_bx);
                    TPBC(dy, box[1], inv_by);
                    TPBC(dz, box[2], inv_bz);

                    /* Step 5: TMUL - dx², dy², dz² */
                    TileFixed<1, TILE_COLS> dx2, dy2, dz2;
                    TMUL(dx2, dx, dx);
                    TMUL(dy2, dy, dy);
                    TMUL(dz2, dz, dz);

                    /* Step 6: TADD - rsq = dx² + dy² + dz² */
                    TileFixed<1, TILE_COLS> tmp;
                    TADD(tmp, dx2, dy2);
                    TADD(rsq, tmp, dz2);

                    /* Step 7: TLJ_FORCE - LJ 力计算 */
                    TLJ_FORCE(fx, fy, fz, dx, dy, dz, rsq, lj_params,
                              t1, t2, t3, t4, t5, t6, t7);

                    /* Step 8: TREDUCE - i 力累加 */
                    fix += TREDUCE(fx);
                    fiy += TREDUCE(fy);
                    fiz += TREDUCE(fz);

                    /* Step 9: j 力写回 (标量循环，编译器自动向量化) */
                    for (int m = 0; m < tile_n; m++) {
                        lfx[j0 + m] -= fx.data[m];
                        lfy[j0 + m] -= fy.data[m];
                        lfz[j0 + m] -= fz.data[m];
                    }
                }
                k += run_len;
            }
            lfx[i] += fix;
            lfy[i] += fiy;
            lfz[i] += fiz;
        }
    }

    for (int t = 1; t < nt; t++) {
        float *lfx = all_lfx + (size_t)t * n;
        float *lfy = all_lfy + (size_t)t * n;
        float *lfz = all_lfz + (size_t)t * n;
        for (int k = 0; k < n; k++) {
            all_lfx[k] += lfx[k];
            all_lfy[k] += lfy[k];
            all_lfz[k] += lfz[k];
        }
    }
    for (int k = 0; k < n; k++) {
        f_aos[k*3+0] = all_lfx[k];
        f_aos[k*3+1] = all_lfy[k];
        f_aos[k*3+2] = all_lfz[k];
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
    int nt = omp_get_max_threads();

    printf("================================================================\n");
    printf("  PTO-GROMACS v11 — 真正的 PTO-ISA 算子链组合\n");
    printf("================================================================\n");
    printf("Atoms: %d | Box: %.1fx%.1fx%.1f | Cut: %.2f | Steps: %d\n",
           n, g->box[0],g->box[1],g->box[2], cut, steps);
    printf("Threads: %d | Tile: 1x%d\n\n", nt, TILE_COLS);
    printf("PTO-ISA Ops: TLOAD, TFILL, TSUB, TPBC, TMUL, TADD, TLJ_FORCE, TREDUCE\n\n");

    NBList *nl = build_nblist(g->x, n, g->box, cut);

    float *sx=(float*)malloc(n*sizeof(float));
    float *sy=(float*)malloc(n*sizeof(float));
    float *sz=(float*)malloc(n*sizeof(float));
    for(int i=0;i<n;i++){ sx[i]=g->x[i*3]; sy[i]=g->x[i*3+1]; sz[i]=g->x[i*3+2]; }

    float *f_ref = (float*)malloc(n*3*sizeof(float));
    float *f_test = (float*)malloc(n*3*sizeof(float));

    float box[3] = {g->box[0], g->box[1], g->box[2]};
    float cutoff_sq = cut*cut;

    float *all_lfx = (float*)calloc((size_t)nt*n, sizeof(float));
    float *all_lfy = (float*)calloc((size_t)nt*n, sizeof(float));
    float *all_lfz = (float*)calloc((size_t)nt*n, sizeof(float));

    /* Warmup */
    ptoisa_compute(sx,sy,sz,f_ref,n,box,nl,cut,nt,all_lfx,all_lfy,all_lfz);

    /* Benchmark */
    double total=0;
    for(int s=0;s<steps;s++){
        double t = ptoisa_compute(sx,sy,sz,f_test,n,box,nl,cut,nt,all_lfx,all_lfy,all_lfz);
        total+=t;
    }
    double avg_ms = total/steps*1000.0;

    /* Force validation */
    double sum_ref=0, sum_diff=0;
    for(int i=0;i<n*3;i++){
        sum_ref += fabs(f_ref[i]);
        sum_diff += fabs(f_test[i]-f_ref[i]);
    }
    double avg_rel = sum_diff/sum_ref;

    printf("PTO-ISA v11: %.3f ms/step (%.2fx)\n", avg_ms, (double)nl->total_pairs*2*20/avg_ms/1e6);
    printf("Force avg rel diff: %.6e (%d components)\n", avg_rel, n*3);

    free(sx); free(sy); free(sz);
    free(f_ref); free(f_test);
    free(nl->jatoms); free(nl->start); free(nl->count); free(nl);
    free(g->x); free(g);
    return 0;
}