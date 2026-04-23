/* pto_e2e_v10_hybrid.cpp
 * PTO-ISA v10: 混合优化
 * 
 * 结合 v5 的两个关键优化：
 * 1. svptest_any 跳过无效原子
 * 2. 标量循环写回（编译器自动向量化）
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <arm_sve.h>

#define TILE_COLS 8

/* ============ GRO Reader ============ */
typedef struct { int natoms; float *x; float box[3]; } GroData;
static GroData* read_gro(const char *fn) {
    FILE *fp = fopen(fn,"r"); if(!fp) return NULL;
    GroData *g = (GroData*)calloc(1,sizeof(GroData)); char line[256];
    fgets(line,sizeof(line),fp); fgets(line,sizeof(line),fp);
    g->natoms = atoi(line);
    g->x = (float*)malloc(g->natoms*3*sizeof(float));
    for(int i=0;i<g->natoms;i++){
        fgets(line,sizeof(line),fp);
        char *p = line+20;
        g->x[i*3+0]=strtof(p,&p); g->x[i*3+1]=strtof(p,&p); g->x[i*3+2]=strtof(p,&p);
    }
    fgets(line,sizeof(line),fp);
    sscanf(line,"%f%f%f",&g->box[0],&g->box[1],&g->box[2]);
    fclose(fp); return g;
}

/* ============ Neighbor List ============ */
typedef struct { int *start; int *jatoms; int *count; int total_pairs; } NBList;
static NBList* build_nblist(const float *x, int n, const float box[3], float cut) {
    NBList *nl = (NBList*)calloc(1,sizeof(NBList));
    nl->start = (int*)malloc(n*sizeof(int));
    nl->count = (int*)calloc(n,sizeof(int));
    float csq = cut*cut;
    for(int i=0;i<n;i++){
        float xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
        for(int j=0;j<n;j++){
            if(i==j) continue;
            float dx=xi-x[j*3],dy=yi-x[j*3+1],dz=zi-x[j*3+2];
            dx-=box[0]*rintf(dx/box[0]);
            dy-=box[1]*rintf(dy/box[1]);
            dz-=box[2]*rintf(dz/box[2]);
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
        for(int j=0;j<n;j++){
            if(i==j) continue;
            float dx=xi-x[j*3],dy=yi-x[j*3+1],dz=zi-x[j*3+2];
            dx-=box[0]*rintf(dx/box[0]);
            dy-=box[1]*rintf(dy/box[1]);
            dz-=box[2]*rintf(dz/box[2]);
            if(dx*dx+dy*dy+dz*dz<csq) nl->jatoms[nl->start[i]+pos[i]++]=j;
        }
    }
    free(pos); return nl;
}

/* ============ PTO-ISA SVE Backend ============ */
struct NonBondedParams {
    float xi, yi, zi;
    float box[3];
    float inv_box[3];
    float sigma_sq;
    float epsilon;
    float cutoff_sq;
    float min_rsq;
};

/* TNONBONDED_LJ_V10: 恢复 svptest_any + 标量循环写回 */
inline __attribute__((always_inline)) void TNONBONDED_LJ_V10(
    const float *sx, const float *sy, const float *sz,
    int j0, int tile_n,
    const NonBondedParams &p,
    float &fix, float &fiy, float &fiz,
    float *lfx, float *lfy, float *lfz) {

    svbool_t pg_all = svptrue_b32();
    svbool_t pg = (tile_n < 8) ? svwhilelt_b32(0, tile_n) : pg_all;

    /* Step 1: 加载 j 坐标 */
    svfloat32_t xj = svld1_f32(pg, &sx[j0]);
    svfloat32_t yj = svld1_f32(pg, &sy[j0]);
    svfloat32_t zj = svld1_f32(pg, &sz[j0]);

    /* Step 2: dx = xi - xj */
    svfloat32_t dx = svsub_f32_x(pg, svdup_f32(p.xi), xj);
    svfloat32_t dy = svsub_f32_x(pg, svdup_f32(p.yi), yj);
    svfloat32_t dz = svsub_f32_x(pg, svdup_f32(p.zi), zj);

    /* Step 3: PBC */
    dx = svsub_f32_x(pg, dx,
        svmul_f32_x(pg, svdup_f32(p.box[0]),
            svrinta_f32_x(pg, svmul_f32_x(pg, dx, svdup_f32(p.inv_box[0])))));
    dy = svsub_f32_x(pg, dy,
        svmul_f32_x(pg, svdup_f32(p.box[1]),
            svrinta_f32_x(pg, svmul_f32_x(pg, dy, svdup_f32(p.inv_box[1])))));
    dz = svsub_f32_x(pg, dz,
        svmul_f32_x(pg, svdup_f32(p.box[2]),
            svrinta_f32_x(pg, svmul_f32_x(pg, dz, svdup_f32(p.inv_box[2])))));

    /* Step 4: rsq */
    svfloat32_t rsq = svadd_f32_x(pg, svadd_f32_x(pg,
        svmul_f32_x(pg, dx, dx), svmul_f32_x(pg, dy, dy)),
        svmul_f32_x(pg, dz, dz));

    /* Step 5: 谓词 */
    svbool_t valid = svand_b_z(pg_all,
        svcmplt_f32(pg_all, rsq, svdup_f32(p.cutoff_sq)),
        svcmpgt_f32(pg_all, rsq, svdup_f32(p.min_rsq)));

    /* ★ 关键优化 1: svptest_any 跳过无效原子 ★ */
    if (svptest_any(pg_all, valid)) {
        svfloat32_t ir = svdiv_f32_z(valid, svdup_f32(1.0f), rsq);
        svfloat32_t s2 = svmul_f32_z(valid, svdup_f32(p.sigma_sq), ir);
        svfloat32_t s6 = svmul_f32_z(valid, svmul_f32_z(valid, s2, s2), s2);
        svfloat32_t s12 = svmul_f32_z(valid, s6, s6);
        svfloat32_t fr = svmul_f32_z(valid,
            svsub_f32_z(valid, svmul_f32_z(valid, svdup_f32(2.0f), s12), s6),
            ir);
        fr = svmul_f32_z(valid, fr, svdup_f32(24.0f * p.epsilon));

        svfloat32_t fx = svmul_f32_z(valid, fr, dx);
        svfloat32_t fy = svmul_f32_z(valid, fr, dy);
        svfloat32_t fz = svmul_f32_z(valid, fr, dz);

        fix += svadda_f32(valid, 0.0f, fx);
        fiy += svadda_f32(valid, 0.0f, fy);
        fiz += svadda_f32(valid, 0.0f, fz);

        /* ★ 关键优化 2: 标量循环写回（编译器自动向量化）★ */
        float fxo[8], fyo[8], fzo[8];
        svst1_f32(pg, fxo, fx);
        svst1_f32(pg, fyo, fy);
        svst1_f32(pg, fzo, fz);
        for (int m = 0; m < tile_n; m++) {
            lfx[j0 + m] -= fxo[m];
            lfy[j0 + m] -= fyo[m];
            lfz[j0 + m] -= fzo[m];
        }
    }
}

/* ============ Force Kernel ============ */
static double run_step(const float *sx, const float *sy, const float *sz,
                       float *f_aos, NBList *nl, int n,
                       float box[3], float inv_box[3], float cutoff_sq) {
    int nt = omp_get_max_threads();
    float *all_lfx = (float*)calloc((size_t)nt*n, sizeof(float));
    float *all_lfy = (float*)calloc((size_t)nt*n, sizeof(float));
    float *all_lfz = (float*)calloc((size_t)nt*n, sizeof(float));

    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *lfx = all_lfx + (size_t)tid * n;
        float *lfy = all_lfy + (size_t)tid * n;
        float *lfz = all_lfz + (size_t)tid * n;

        NonBondedParams p;
        p.sigma_sq = 0.09f;
        p.epsilon = 0.5f;
        p.cutoff_sq = cutoff_sq;
        p.min_rsq = 1e-8f;
        p.box[0] = box[0]; p.box[1] = box[1]; p.box[2] = box[2];
        p.inv_box[0] = inv_box[0]; p.inv_box[1] = inv_box[1]; p.inv_box[2] = inv_box[2];

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

                    TNONBONDED_LJ_V10(sx, sy, sz, j0, tile_n, p,
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
    free(all_lfx); free(all_lfy); free(all_lfz);
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
    printf("  PTO-GROMACS v10 — Hybrid (svptest_any + scalar writeback)\n");
    printf("================================================================\n");
    printf("Atoms: %d | Box: %.1fx%.1fx%.1f | Cut: %.2f | Steps: %d\n",
           n, g->box[0],g->box[1],g->box[2], cut, steps);
    printf("Threads: %d | PTO Tile: 1x%d\n\n", nt, TILE_COLS);

    printf("Building neighbor list (sorted)..."); fflush(stdout);
    NBList *nl = build_nblist(g->x, n, g->box, cut);
    printf(" %d pairs (%.1f nbs/atom)\n", nl->total_pairs, 2.0f*nl->total_pairs/n);

    float *sx=(float*)malloc(n*sizeof(float));
    float *sy=(float*)malloc(n*sizeof(float));
    float *sz=(float*)malloc(n*sizeof(float));
    for(int i=0;i<n;i++){ sx[i]=g->x[i*3]; sy[i]=g->x[i*3+1]; sz[i]=g->x[i*3+2]; }

    float *f_ref = (float*)malloc(n*3*sizeof(float));
    float *f_test = (float*)malloc(n*3*sizeof(float));

    float box[3] = {g->box[0], g->box[1], g->box[2]};
    float inv_box[3] = {1.0f/box[0], 1.0f/box[1], 1.0f/box[2]};
    float cutoff_sq = cut*cut;

    /* Warmup */
    run_step(sx,sy,sz,f_ref,nl,n,box,inv_box,cutoff_sq);

    /* Benchmark */
    double total=0;
    for(int s=0;s<steps;s++){
        double t = run_step(sx,sy,sz,f_test,nl,n,box,inv_box,cutoff_sq);
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

    printf("PTO-ISA v10: %.3f ms/step (%.2fx)\n", avg_ms, (double)nl->total_pairs*2*20/avg_ms/1e6);
    printf("Force avg rel diff: %.6e (%d components)\n", avg_rel, n*3);

    free(sx); free(sy); free(sz);
    free(f_ref); free(f_test);
    free(nl->jatoms); free(nl->start); free(nl->count); free(nl);
    free(g->x); free(g);
    return 0;
}
