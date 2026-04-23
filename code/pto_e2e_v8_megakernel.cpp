/**
 * PTO-GROMACS v6 — PTO-ISA 算子融合实现
 *
 * 使用 PTO-ISA 核心算子 (pto_gromacs_core.hpp) 重写
 * GROMACS non-bonded LJ force kernel。
 *
 * PTO-ISA 算子融合链:
 *   TLOAD(j坐标) → TSUB(距离) → TPBC(周期边界)
 *   → TMUL+TADD(rsq) → TLJ_FORCE(ir→s2→s6→s12→fr→fx/fy/fz)
 *   → TROWTSUM(i力) + TSCATTER(j力)
 *
 * 编译: g++ -O3 -march=armv8.6-a+sve -msve-vector-bits=256
 *        -ffast-math -fopenmp -std=c++17 -I. pto_e2e_v6_ptoisa.cpp -lm -fopenmp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "pto_gromacs_core.hpp"

using namespace pto::cpu;

constexpr int TILE_COLS = 8; /* SVE 256-bit = 8 floats */

/* ============ GRO parser ============ */
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

/* ============ Neighbor list (sorted) ============ */
typedef struct { int *start; int *jatoms; int *count; int total_pairs; } NBList;
static NBList* build_nblist(const float *x, int n, const float box[3], float cut) {
    NBList *nl = (NBList*)calloc(1,sizeof(NBList));
    nl->start = (int*)malloc(n*sizeof(int));
    nl->count = (int*)calloc(n,sizeof(int));
    float csq = cut*cut;
    for(int i=0;i<n;i++){
        float xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2]; int cnt=0;
        for(int j=i+1;j<n;j++){
            float dx=xi-x[j*3],dy=yi-x[j*3+1],dz=zi-x[j*3+2];
            dx-=box[0]*rintf(dx/box[0]); dy-=box[1]*rintf(dy/box[1]); dz-=box[2]*rintf(dz/box[2]);
            if(dx*dx+dy*dy+dz*dz<csq) cnt++;
        }
        nl->count[i]=cnt; nl->total_pairs+=cnt;
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
 * PTO-ISA LJ Force Kernel — 算子融合实现
 *
 * 每个 i 原子的 j 邻居按排序后的连续 run 处理。
 * 每个 run 按 TILE_COLS 粒度切分为 PTO Tile 操作。
 * 所有中间结果 (dx, rsq, ir, s2, s6, s12, fr, fx/fy/fz)
 * 均在 Tile (栈数组) 中, 不写回全局内存 → 算子融合。
 * ==================================================================== */
static double ptoisa_nb_compute(float *sx, float *sy, float *sz,
                                 float *f_aos, int n, float box[3],
                                 NBList *nl, float cut, int nt,
                                 float *all_lfx, float *all_lfy, float *all_lfz) {
    LJParamsT<1, TILE_COLS> lj_params;
    lj_params.sigma_sq = 0.09f;
    lj_params.epsilon = 0.5f;
    lj_params.cutoff_sq = cut * cut;
    lj_params.min_rsq = 1e-8f;
    float inv_bx = 1.0f / box[0], inv_by = 1.0f / box[1], inv_bz = 1.0f / box[2];

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

        /* ===== PTO 超级融合算子参数 ===== */
        NonBondedParams p;
        p.sigma_sq = 0.09f;
        p.epsilon = 0.5f;
        p.cutoff_sq = lj_params.cutoff_sq;
        p.min_rsq = 1e-8f;
        p.box[0] = box[0]; p.box[1] = box[1]; p.box[2] = box[2];
        p.inv_box[0] = inv_bx; p.inv_box[1] = inv_by; p.inv_box[2] = inv_bz;

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < n; i++) {
            p.xi = sx[i]; p.yi = sy[i]; p.zi = sz[i];
            float fix = 0, fiy = 0, fiz = 0;
            int ni = nl->count[i];
            int base = nl->start[i];

            int k = 0;
            while (k < ni) {
                int j_start = nl->jatoms[base + k];
                int run_len = 1;
                while (k + run_len < ni &&
                       nl->jatoms[base + k + run_len] == j_start + run_len)
                    run_len++;

                for (int r = 0; r < run_len; r += TILE_COLS) {
                    int rem = run_len - r;
                    int tile_n = (rem < TILE_COLS) ? rem : TILE_COLS;
                    int j0 = j_start + r;

                    /* ★ TNONBONDED_LJ: 超级融合算子 (向量化j力写回) ★ */
                    TNONBONDED_LJ(sx, sy, sz, j0, tile_n, p,
                                  fix, fiy, fiz, lfx, lfy, lfz);
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

/* ============ Main ============ */
int main(int argc, char *argv[]) {
    if(argc<2){ printf("Usage: %s <file.gro> [cutoff] [nsteps]\n",argv[0]); return 1; }
    float cut = argc>2?atof(argv[2]):1.0f;
    int steps = argc>3?atoi(argv[3]):200;

    GroData *g = read_gro(argv[1]);
    if(!g) { fprintf(stderr,"Cannot read %s\n",argv[1]); return 1; }
    int n = g->natoms;
    int nt = omp_get_max_threads();

    printf("================================================================\n");
    printf("  PTO-GROMACS v8 — TNONBONDED_LJ Mega-Kernel (SVE Zero-Spill)\n");
    printf("================================================================\n");
    printf("Atoms: %d | Box: %.1fx%.1fx%.1f | Cut: %.2f | Steps: %d\n",
           n, g->box[0],g->box[1],g->box[2], cut, steps);
    printf("Threads: %d | PTO Tile: 1x%d | LJ sigma_sq=%.3f eps=%.2f\n",
           nt, TILE_COLS, 0.09f, 0.5f);
    printf("PTO Ops: TLOAD, TFILL, TSUB, TPBC, TMUL, TADD, TLJ_FORCE, TROWTSUM, TSCATTER\n\n");

    printf("Building neighbor list (sorted)..."); fflush(stdout);
    NBList *nl = build_nblist(g->x, n, g->box, cut);
    printf(" %d pairs (%.1f nbs/atom)\n", nl->total_pairs, 2.0f*nl->total_pairs/n);

    float *sx=(float*)malloc(n*sizeof(float));
    float *sy=(float*)malloc(n*sizeof(float));
    float *sz=(float*)malloc(n*sizeof(float));
    for(int i=0;i<n;i++){ sx[i]=g->x[i*3]; sy[i]=g->x[i*3+1]; sz[i]=g->x[i*3+2]; }

    float *all_lfx=(float*)calloc((size_t)nt*n,sizeof(float));
    float *all_lfy=(float*)calloc((size_t)nt*n,sizeof(float));
    float *all_lfz=(float*)calloc((size_t)nt*n,sizeof(float));
    float *f_scalar=(float*)malloc(n*3*sizeof(float));
    float *f_pto=(float*)malloc(n*3*sizeof(float));

    printf("Scalar:      "); fflush(stdout);
    scalar_nb(g->x, f_scalar, n, g->box, nl, cut);
    double ts=0;
    for(int s=0;s<steps;s++) ts+=scalar_nb(g->x, f_scalar, n, g->box, nl, cut);
    printf("%.3f ms/step\n", ts/steps*1000);

    printf("PTO-ISA v6:  "); fflush(stdout);
    ptoisa_nb_compute(sx, sy, sz, f_pto, n, g->box, nl, cut, nt, all_lfx, all_lfy, all_lfz);
    double tp=0;
    for(int s=0;s<steps;s++) tp+=ptoisa_nb_compute(sx, sy, sz, f_pto, n, g->box, nl, cut, nt, all_lfx, all_lfy, all_lfz);
    printf("%.3f ms/step (%.2fx)\n", tp/steps*1000, ts/tp);

    float max_diff=0, rel_diff_sum=0; int diff_count=0;
    for(int k=0;k<n*3;k++){
        float d=fabsf(f_scalar[k]-f_pto[k]);
        if(d>max_diff) max_diff=d;
        if(fabsf(f_scalar[k])>1e-6f){ rel_diff_sum+=d/fabsf(f_scalar[k]); diff_count++; }
    }

    printf("\n================================================================\n");
    printf("  Performance Report\n");
    printf("================================================================\n");
    printf("Scalar:    %.3f ms/step (1.00x)\n", ts/steps*1000);
    printf("PTO-ISA:   %.3f ms/step (%.2fx)\n", tp/steps*1000, ts/tp);
    printf("Force max diff: %.6e\n", max_diff);
    printf("Force avg rel diff: %.6e (%d components)\n",
           diff_count>0 ? rel_diff_sum/diff_count : 0, diff_count);
    printf("================================================================\n");

    free(sx); free(sy); free(sz);
    free(all_lfx); free(all_lfy); free(all_lfz);
    free(f_scalar); free(f_pto);
    free(nl->start); free(nl->jatoms); free(nl->count); free(nl);
    free(g->x); free(g);
    return 0;
}
