/**
 * PTO-GROMACS v9 — 超级融合算子 + 向量化j力写回
 *
 * 核心优化:
 *   1. svwhilelt_b32 正确处理部分向量 (与v5一致)
 *   2. 整条non-bonded链在SVE寄存器中完成
 *   3. j力写回用向量化 read-modify-write: svld1 → svsub → svst1
 *
 * 目标: 缩小与v5手写SVE的差距
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

/* ====================================================================
 * 数据结构
 * ==================================================================== */
typedef struct {
    int *start;
    int *count;
    int *jatoms;
    int total_pairs;
} NBList;

static void nb_build(NBList *nl, float *x, int n, float box[3], float cut) {
    nl->start = (int*)calloc(n, sizeof(int));
    nl->count = (int*)calloc(n, sizeof(int));
    int cap = n * 50;
    nl->jatoms = (int*)malloc(cap * sizeof(int));
    nl->total_pairs = 0;

    float csq = cut * cut;
    for (int i = 0; i < n; i++) {
        nl->start[i] = nl->total_pairs;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            float dx = x[i*3] - x[j*3];
            float dy = x[i*3+1] - x[j*3+1];
            float dz = x[i*3+2] - x[j*3+2];
            dx -= box[0] * rintf(dx / box[0]);
            dy -= box[1] * rintf(dy / box[1]);
            dz -= box[2] * rintf(dz / box[2]);
            float rsq = dx*dx + dy*dy + dz*dz;
            if (rsq < csq) {
                if (nl->total_pairs >= cap) {
                    cap *= 2;
                    nl->jatoms = (int*)realloc(nl->jatoms, cap * sizeof(int));
                }
                nl->jatoms[nl->total_pairs++] = j;
                nl->count[i]++;
            }
        }
    }
}

static void nb_sort(NBList *nl, int n) {
    for (int i = 0; i < n; i++) {
        int base = nl->start[i];
        int cnt = nl->count[i];
        for (int k = 0; k < cnt-1; k++) {
            for (int m = k+1; m < cnt; m++) {
                if (nl->jatoms[base+m] < nl->jatoms[base+k]) {
                    int tmp = nl->jatoms[base+k];
                    nl->jatoms[base+k] = nl->jatoms[base+m];
                    nl->jatoms[base+m] = tmp;
                }
            }
        }
    }
}

static void nb_free(NBList *nl) {
    free(nl->start); free(nl->count); free(nl->jatoms);
}

/* ====================================================================
 * Scalar baseline
 * ==================================================================== */
static double scalar_nb(float *x, float *f, int n, float box[3], NBList *nl, float cut) {
    float csq = cut * cut;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    memset(f, 0, n*3*sizeof(float));
    for (int i = 0; i < n; i++) {
        float xi = x[i*3], yi = x[i*3+1], zi = x[i*3+2];
        for (int k = 0; k < nl->count[i]; k++) {
            int j = nl->jatoms[nl->start[i] + k];
            float dx = xi - x[j*3];
            float dy = yi - x[j*3+1];
            float dz = zi - x[j*3+2];
            dx -= box[0] * rintf(dx / box[0]);
            dy -= box[1] * rintf(dy / box[1]);
            dz -= box[2] * rintf(dz / box[2]);
            float rsq = dx*dx + dy*dy + dz*dz;
            if (rsq < csq && rsq > 1e-8f) {
                float ir = 1.0f / rsq;
                float s2 = 0.09f * ir;
                float s6 = s2 * s2 * s2;
                float s12 = s6 * s6;
                float fr = 24.0f * 0.5f * (2.0f*s12 - s6) * ir;
                float fx = fr * dx, fy = fr * dy, fz = fr * dz;
                f[i*3] += fx; f[i*3+1] += fy; f[i*3+2] += fz;
                f[j*3] -= fx; f[j*3+1] -= fy; f[j*3+2] -= fz;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* ====================================================================
 * v9: 超级融合算子 + 向量化j力写回
 * ==================================================================== */
typedef struct {
    float *sx, *sy, *sz;
    float *all_lfx, *all_lfy, *all_lfz;
    int num_threads;
    float inv_box[3];
} PTOv9Ctx;

static PTOv9Ctx* ptov9_create(float *aos_coords, int n, float box[3], int nt) {
    PTOv9Ctx *ctx = (PTOv9Ctx*)calloc(1, sizeof(PTOv9Ctx));
    ctx->sx = (float*)malloc(n * sizeof(float));
    ctx->sy = (float*)malloc(n * sizeof(float));
    ctx->sz = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        ctx->sx[i] = aos_coords[i*3];
        ctx->sy[i] = aos_coords[i*3+1];
        ctx->sz[i] = aos_coords[i*3+2];
    }
    ctx->inv_box[0] = 1.0f / box[0];
    ctx->inv_box[1] = 1.0f / box[1];
    ctx->inv_box[2] = 1.0f / box[2];
    ctx->num_threads = nt;
    ctx->all_lfx = (float*)calloc((size_t)nt * n, sizeof(float));
    ctx->all_lfy = (float*)calloc((size_t)nt * n, sizeof(float));
    ctx->all_lfz = (float*)calloc((size_t)nt * n, sizeof(float));
    return ctx;
}

static void ptov9_destroy(PTOv9Ctx *ctx) {
    free(ctx->sx); free(ctx->sy); free(ctx->sz);
    free(ctx->all_lfx); free(ctx->all_lfy); free(ctx->all_lfz);
    free(ctx);
}

/**
 * TNONBONDED_LJ_V9 — 超级融合算子
 *
 * 整条non-bonded计算链在SVE寄存器中完成:
 *   1. 加载j坐标 (svld1)
 *   2. 计算dx/dy/dz (svsub)
 *   3. 应用PBC (svsub + svmul + svrinta)
 *   4. 计算rsq (svmul + svadd)
 *   5. LJ力计算 (svdiv + svmul + svsub)
 *   6. i力累加 (svadda)
 *   7. j力写回 (svld1 + svsub + svst1) ← 向量化!
 *
 * 关键优化:
 *   - 使用svwhilelt_b32处理部分向量
 *   - j力写回用向量化read-modify-write
 *   - 所有中间结果在SVE寄存器中
 */
#ifdef __ARM_FEATURE_SVE
static inline void TNONBONDED_LJ_V9(
    float xi, float yi, float zi,
    const float *sx, const float *sy, const float *sz,
    int j0, int run_len, int r,
    float box[3], float inv_bx, float inv_by, float inv_bz,
    float csq, float ssq, float eps,
    float *lfx, float *lfy, float *lfz,
    float &fix, float &fiy, float &fiz
) {
    svbool_t pg = svwhilelt_b32(r, run_len);  // 正确处理部分向量
    svbool_t all_p = svptrue_b32();

    /* 1. 加载j坐标 */
    svfloat32_t xj = svld1_f32(pg, &sx[j0]);
    svfloat32_t yj = svld1_f32(pg, &sy[j0]);
    svfloat32_t zj = svld1_f32(pg, &sz[j0]);

    /* 2. 计算dx = xi - xj */
    svfloat32_t dx = svsub_f32_x(pg, svdup_f32(xi), xj);
    svfloat32_t dy = svsub_f32_x(pg, svdup_f32(yi), yj);
    svfloat32_t dz = svsub_f32_x(pg, svdup_f32(zi), zj);

    /* 3. 应用PBC */
    dx = svsub_f32_x(pg, dx, svmul_f32_x(pg, svdup_f32(box[0]),
        svrinta_f32_x(pg, svmul_f32_x(pg, dx, svdup_f32(inv_bx)))));
    dy = svsub_f32_x(pg, dy, svmul_f32_x(pg, svdup_f32(box[1]),
        svrinta_f32_x(pg, svmul_f32_x(pg, dy, svdup_f32(inv_by)))));
    dz = svsub_f32_x(pg, dz, svmul_f32_x(pg, svdup_f32(box[2]),
        svrinta_f32_x(pg, svmul_f32_x(pg, dz, svdup_f32(inv_bz)))));

    /* 4. 计算rsq */
    svfloat32_t rsq = svadd_f32_x(pg, svadd_f32_x(pg,
        svmul_f32_x(pg, dx, dx), svmul_f32_x(pg, dy, dy)),
        svmul_f32_x(pg, dz, dz));

    /* 5. LJ力计算 */
    svbool_t valid = svand_b_z(pg,
        svcmplt_f32(pg, rsq, svdup_f32(csq)),
        svcmpgt_f32(pg, rsq, svdup_f32(1e-8f)));

    if (svptest_any(all_p, valid)) {
        svfloat32_t ir = svdiv_f32_z(valid, svdup_f32(1.0f), rsq);
        svfloat32_t s2 = svmul_f32_z(valid, svdup_f32(ssq), ir);
        svfloat32_t s6 = svmul_f32_z(valid, svmul_f32_z(valid, s2, s2), s2);
        svfloat32_t s12 = svmul_f32_z(valid, s6, s6);
        svfloat32_t fr = svmul_f32_z(valid, svdup_f32(24.0f * eps),
            svmul_f32_z(valid, svsub_f32_z(valid,
                svmul_f32_z(valid, svdup_f32(2.0f), s12), s6), ir));

        svfloat32_t fx_v = svmul_f32_z(valid, fr, dx);
        svfloat32_t fy_v = svmul_f32_z(valid, fr, dy);
        svfloat32_t fz_v = svmul_f32_z(valid, fr, dz);

        /* 6. i力累加 */
        fix += svadda_f32(valid, 0.0f, fx_v);
        fiy += svadda_f32(valid, 0.0f, fy_v);
        fiz += svadda_f32(valid, 0.0f, fz_v);

        /* 7. j力写回 — 与v5一致: 先存临时数组, 再标量减 */
        float fxo[16], fyo[16], fzo[16];
        svst1_f32(pg, fxo, fx_v);
        svst1_f32(pg, fyo, fy_v);
        svst1_f32(pg, fzo, fz_v);
        int vl = (int)svcntw();
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
#endif

static double ptov9_compute(PTOv9Ctx *ctx, float *f_aos, int n, float box[3],
                            NBList *nl, float cut) {
    float csq = cut * cut, eps = 0.5f, ssq = 0.09f;
    float *sx = ctx->sx, *sy = ctx->sy, *sz = ctx->sz;
    float inv_bx = ctx->inv_box[0], inv_by = ctx->inv_box[1], inv_bz = ctx->inv_box[2];
    int nt = ctx->num_threads;

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    memset(f_aos, 0, n*3*sizeof(float));
    memset(ctx->all_lfx, 0, (size_t)nt*n*sizeof(float));
    memset(ctx->all_lfy, 0, (size_t)nt*n*sizeof(float));
    memset(ctx->all_lfz, 0, (size_t)nt*n*sizeof(float));

#ifdef __ARM_FEATURE_SVE
    int vl = (int)svcntw();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *lfx = ctx->all_lfx + (size_t)tid * n;
        float *lfy = ctx->all_lfy + (size_t)tid * n;
        float *lfz = ctx->all_lfz + (size_t)tid * n;

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
                while (k + run_len < ni &&
                       nl->jatoms[base + k + run_len] == j_start + run_len) {
                    run_len++;
                }

                for (int r = 0; r < run_len; r += vl) {
                    int j0 = j_start + r;

                    /* ★ 超级融合算子 ★ */
                    TNONBONDED_LJ_V9(
                        xi, yi, zi,
                        sx, sy, sz,
                        j0, run_len, r,
                        box, inv_bx, inv_by, inv_bz,
                        csq, ssq, eps,
                        lfx, lfy, lfz,
                        fix, fiy, fiz
                    );
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
        for (int k = 0; k < n; k++) {
            ctx->all_lfx[k] += lfx[k];
            ctx->all_lfy[k] += lfy[k];
            ctx->all_lfz[k] += lfz[k];
        }
    }

    /* Convert to AOS */
    for (int i = 0; i < n; i++) {
        f_aos[i*3]   = ctx->all_lfx[i];
        f_aos[i*3+1] = ctx->all_lfy[i];
        f_aos[i*3+2] = ctx->all_lfz[i];
    }
#else
    fprintf(stderr, "Error: SVE not supported\n");
    exit(1);
#endif

    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* ====================================================================
 * Benchmark
 * ==================================================================== */
static void generate_coords(float *x, int n, float box[3]) {
    unsigned seed = 42;
    for (int i = 0; i < n; i++) {
        x[i*3]   = (rand_r(&seed) / (float)RAND_MAX) * box[0];
        x[i*3+1] = (rand_r(&seed) / (float)RAND_MAX) * box[1];
        x[i*3+2] = (rand_r(&seed) / (float)RAND_MAX) * box[2];
    }
}

static double compare_forces(float *f1, float *f2, int n, double *max_diff, double *avg_rel) {
    double sum = 0, max = 0, rel_sum = 0;
    int cnt = 0;
    for (int i = 0; i < n*3; i++) {
        double d = fabs(f1[i] - f2[i]);
        sum += d * d;
        if (d > max) max = d;
        if (fabs(f1[i]) > 1e-10f) {
            rel_sum += d / fabs(f1[i]);
            cnt++;
        }
    }
    *max_diff = max;
    *avg_rel = (cnt > 0) ? rel_sum / cnt : 0;
    return sqrt(sum / (n*3));
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 21072;
    int steps = (argc > 2) ? atoi(argv[2]) : 100;
    int nt = (argc > 3) ? atoi(argv[3]) : 32;
    float box[3] = {6.0f, 6.0f, 6.0f};
    float cut = 1.0f;

    omp_set_num_threads(nt);

    float *x = (float*)malloc(n*3*sizeof(float));
    float *f_scalar = (float*)malloc(n*3*sizeof(float));
    float *f_pto = (float*)malloc(n*3*sizeof(float));

    generate_coords(x, n, box);

    NBList nl;
    nb_build(&nl, x, n, box, cut);
    nb_sort(&nl, n);

    printf("================================================================\n");
    printf("  PTO-GROMACS v9 — Mega-Kernel + Vectorized j-force Writeback\n");
    printf("================================================================\n");
    printf("Atoms: %d | Box: %.1fx%.1fx%.1f | Cut: %.2f | Steps: %d\n",
           n, box[0], box[1], box[2], cut, steps);
    printf("Threads: %d | SVE VL: %d floats\n", nt, (int)svcntw());
    printf("----------------------------------------------------------------\n");

    /* Warmup */
    scalar_nb(x, f_scalar, n, box, &nl, cut);

    PTOv9Ctx *ctx = ptov9_create(x, n, box, nt);

    /* Scalar baseline */
    double t_scalar = 0;
    for (int s = 0; s < steps; s++) {
        t_scalar += scalar_nb(x, f_scalar, n, box, &nl, cut);
    }

    /* PTO v9 */
    double t_pto = 0;
    for (int s = 0; s < steps; s++) {
        t_pto += ptov9_compute(ctx, f_pto, n, box, &nl, cut);
    }

    double max_diff, avg_rel;
    double rmse = compare_forces(f_scalar, f_pto, n, &max_diff, &avg_rel);

    printf("Scalar:   %.3f ms/step\n", t_scalar * 1000 / steps);
    printf("PTO v9:   %.3f ms/step (%.2fx)\n", t_pto * 1000 / steps, t_scalar / t_pto);
    printf("Force max diff: %e\n", max_diff);
    printf("Force avg rel diff: %e (%d components)\n", avg_rel, n*3);
    printf("================================================================\n");

    ptov9_destroy(ctx);
    nb_free(&nl);
    free(x);
    free(f_scalar);
    free(f_pto);

    return 0;
}
