/*
 * PTO-GROMACS 端到端性能对比测试
 * 
 * 读取GROMACS .gro文件，模拟完整MD步骤中的非键力计算
 * 对比：GROMACS风格标量基线 vs PTO-SVE算子融合优化
 *
 * 编译:
 *   gcc -O3 -march=armv9-a+sve+sve2 -msve-vector-bits=256 -ffast-math -fopenmp \
 *       pto_gromacs_e2e_benchmark.c gromacs_pto_tiling.o gromacs_pto_sve.o gromacs_pto_sme.o \
 *       -o pto_e2e_benchmark -lm -fopenmp
 *
 * 运行:
 *   OMP_NUM_THREADS=16 ./pto_e2e_benchmark ../../gromacs_benchmark/em_medium.gro 1.0
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
    float *v;       /* velocities */
    float box[3];   /* box dimensions nm */
} GroData;

static GroData* read_gro(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return NULL; }
    
    GroData *gro = calloc(1, sizeof(GroData));
    char line[256];
    
    /* title line */
    fgets(line, sizeof(line), fp);
    /* atom count */
    fgets(line, sizeof(line), fp);
    gro->natoms = atoi(line);
    
    gro->x = malloc(gro->natoms * 3 * sizeof(float));
    gro->v = calloc(gro->natoms * 3, sizeof(float));
    
    for (int i = 0; i < gro->natoms; i++) {
        fgets(line, sizeof(line), fp);
        /* gro format: %5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f */
        char *p = line + 20; /* skip resid(5) resname(5) atomname(5) atomid(5) */
        gro->x[i*3+0] = strtof(p, &p);  /* x, nm */
        gro->x[i*3+1] = strtof(p, &p);  /* y */
        gro->x[i*3+2] = strtof(p, &p);  /* z */
    }
    
    /* box line */
    fgets(line, sizeof(line), fp);
    sscanf(line, "%f %f %f", &gro->box[0], &gro->box[1], &gro->box[2]);
    
    fclose(fp);
    return gro;
}

static void free_gro(GroData *gro) {
    if (gro) { free(gro->x); free(gro->v); free(gro); }
}

/* ============ 邻居列表构建 (GROMACS风格Verlet邻居列表) ============ */

typedef struct {
    int *neighbors;    /* neighbors[i] = start index into neighbor_atoms */
    int *neighbor_atoms;/* j-atom indices */
    int *count;        /* count[i] = number of neighbors of atom i */
    int total_pairs;
} NeighborList;

static NeighborList* build_neighbor_list(float *x, int natoms, float box[3], float cutoff) {
    NeighborList *nl = calloc(1, sizeof(NeighborList));
    nl->neighbors = malloc(natoms * sizeof(int));
    nl->count = calloc(natoms, sizeof(int));
    
    /* First pass: count neighbors */
    float cutoff_sq = cutoff * cutoff;
    int total = 0;
    for (int i = 0; i < natoms; i++) {
        for (int j = i + 1; j < natoms; j++) {
            float dx = x[i*3+0] - x[j*3+0];
            float dy = x[i*3+1] - x[j*3+1];
            float dz = x[i*3+2] - x[j*3+2];
            /* Minimum image convention */
            dx -= box[0] * roundf(dx / box[0]);
            dy -= box[1] * roundf(dy / box[1]);
            dz -= box[2] * roundf(dz / box[2]);
            float rsq = dx*dx + dy*dy + dz*dz;
            if (rsq < cutoff_sq) {
                nl->count[i]++;
                total++;
            }
        }
    }
    
    /* Build prefix sum */
    nl->neighbors[0] = 0;
    for (int i = 1; i < natoms; i++) {
        nl->neighbors[i] = nl->neighbors[i-1] + nl->count[i-1];
    }
    nl->neighbor_atoms = malloc(total * sizeof(int));
    nl->total_pairs = total;
    
    /* Second pass: fill neighbors */
    int *pos = calloc(natoms, sizeof(int));
    for (int i = 0; i < natoms; i++) {
        for (int j = i + 1; j < natoms; j++) {
            float dx = x[i*3+0] - x[j*3+0];
            float dy = x[i*3+1] - x[j*3+1];
            float dz = x[i*3+2] - x[j*3+2];
            dx -= box[0] * roundf(dx / box[0]);
            dy -= box[1] * roundf(dy / box[1]);
            dz -= box[2] * roundf(dz / box[2]);
            float rsq = dx*dx + dy*dy + dz*dz;
            if (rsq < cutoff_sq) {
                nl->neighbor_atoms[nl->neighbors[i] + pos[i]] = j;
                pos[i]++;
            }
        }
    }
    free(pos);
    return nl;
}

static void free_neighbor_list(NeighborList *nl) {
    if (nl) { free(nl->neighbors); free(nl->neighbor_atoms); free(nl->count); free(nl); }
}

/* ============ 标量基线: GROMACS风格非键力计算 ============ */

static double compute_forces_scalar(float *x, float *f, int natoms, float box[3],
                                    NeighborList *nl, float cutoff) {
    float cutoff_sq = cutoff * cutoff;
    float epsilon = 0.5f, sigma = 0.3f;  /* LJ params (SPC water O-O) */
    float sigma_sq = sigma * sigma;
    
    double t0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t0 = ts.tv_sec + ts.tv_nsec * 1e-9;
    
    memset(f, 0, natoms * 3 * sizeof(float));
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < natoms; i++) {
        float fx_i = 0, fy_i = 0, fz_i = 0;
        float xi = x[i*3+0], yi = x[i*3+1], zi = x[i*3+2];
        
        for (int n = 0; n < nl->count[i]; n++) {
            int j = nl->neighbor_atoms[nl->neighbors[i] + n];
            
            float dx = xi - x[j*3+0];
            float dy = yi - x[j*3+1];
            float dz = zi - x[j*3+2];
            dx -= box[0] * roundf(dx / box[0]);
            dy -= box[1] * roundf(dy / box[1]);
            dz -= box[2] * roundf(dz / box[2]);
            
            float rsq = dx*dx + dy*dy + dz*dz;
            if (rsq < cutoff_sq && rsq > 1e-8f) {
                float inv_rsq = 1.0f / rsq;
                float s2 = sigma_sq * inv_rsq;
                float s6 = s2*s2*s2;
                float s12 = s6*s6;
                
                /* LJ force: 24*eps*(2*s12 - s6)/r^2 */
                float f_over_r = 24.0f * epsilon * (2.0f*s12 - s6) * inv_rsq;
                
                fx_i += f_over_r * dx;
                fy_i += f_over_r * dy;
                fz_i += f_over_r * dz;
                
                #pragma omp atomic
                f[j*3+0] -= f_over_r * dx;
                #pragma omp atomic
                f[j*3+1] -= f_over_r * dy;
                #pragma omp atomic
                f[j*3+2] -= f_over_r * dz;
            }
        }
        
        #pragma omp atomic
        f[i*3+0] += fx_i;
        #pragma omp atomic
        f[i*3+1] += fy_i;
        #pragma omp atomic
        f[i*3+2] += fz_i;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec + ts.tv_nsec * 1e-9) - t0;
}

/* ============ PTO-SVE优化版: 算子融合非键力计算 ============ */

static double compute_forces_pto_sve(float *x, float *f, int natoms, float box[3],
                                      NeighborList *nl, float cutoff) {
    float cutoff_sq = cutoff * cutoff;
    float epsilon = 0.5f, sigma = 0.3f;
    float sigma_sq = sigma * sigma;
    
    double t0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t0 = ts.tv_sec + ts.tv_nsec * 1e-9;
    
    memset(f, 0, natoms * 3 * sizeof(float));
    
    svbool_t all_p = svptrue_b32();
    int vl = svcntw(); /* SVE vector length in floats */
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < natoms; i++) {
        float fx_i = 0, fy_i = 0, fz_i = 0;
        
        /* Broadcast i-atom coordinates */
        svfloat32_t xi_v = svdup_f32(x[i*3+0]);
        svfloat32_t yi_v = svdup_f32(x[i*3+1]);
        svfloat32_t zi_v = svdup_f32(x[i*3+2]);
        
        /* Accumulators in registers - no intermediate memory writes */
        svfloat32_t fx_acc = svdup_f32(0.0f);
        svfloat32_t fy_acc = svdup_f32(0.0f);
        svfloat32_t fz_acc = svdup_f32(0.0f);
        
        int nn = nl->count[i];
        const int *j_atoms = &nl->neighbor_atoms[nl->neighbors[i]];
        
        /* Process j-atoms in SVE vector-width chunks */
        for (int k = 0; k < nn; k += vl) {
            int remaining = nn - k;
            svbool_t pg = svwhilelt_b32(k, nn);
            
            /* Gather j-atom coordinates (scalar gather for non-contiguous) */
            svfloat32_t jx_v, jy_v, jz_v;
            float jx_buf[vl], jy_buf[vl], jz_buf[vl];
            for (int m = 0; m < remaining && m < vl; m++) {
                int j = j_atoms[k + m];
                jx_buf[m] = x[j*3+0];
                jy_buf[m] = x[j*3+1];
                jz_buf[m] = x[j*3+2];
            }
            jx_v = svld1_f32(pg, jx_buf);
            jy_v = svld1_f32(pg, jy_buf);
            jz_v = svld1_f32(pg, jz_buf);
            
            /* Fused: distance + cutoff + LJ + force, all in registers */
            svfloat32_t dx = svsub_f32_x(pg, xi_v, jx_v);
            svfloat32_t dy = svsub_f32_x(pg, yi_v, jy_v);
            svfloat32_t dz = svsub_f32_x(pg, zi_v, jz_v);
            
            svfloat32_t rsq = svadd_f32_x(pg, svadd_f32_x(pg, 
                svmul_f32_x(pg, dx, dx), svmul_f32_x(pg, dy, dy)),
                svmul_f32_x(pg, dz, dz));
            
            /* Cutoff mask */
            svbool_t in_cut = svcmplt_f32(pg, rsq, svdup_f32(cutoff_sq));
            svbool_t valid = svand_b_z(pg, in_cut, svcmpgt_f32(pg, rsq, svdup_f32(1e-8f)));
            
            if (svptest_any(all_p, valid)) {
                /* Fused LJ computation - all in registers */
                svfloat32_t inv_rsq = svdiv_f32_z(valid, svdup_f32(1.0f), rsq);
                svfloat32_t s2 = svmul_f32_z(valid, svdup_f32(sigma_sq), inv_rsq);
                svfloat32_t s6 = svmul_f32_z(valid, svmul_f32_z(valid, s2, s2), s2);
                svfloat32_t s12 = svmul_f32_z(valid, s6, s6);
                
                /* f/r = 24*eps*(2*s12 - s6) / r^2 */
                svfloat32_t f_over_r = svmul_f32_z(valid, svdup_f32(24.0f * epsilon),
                    svmul_f32_z(valid, svsub_f32_z(valid, svmul_f32_z(valid, svdup_f32(2.0f), s12), s6),
                    inv_rsq));
                
                /* Force components */
                svfloat32_t fx_v = svmul_f32_z(valid, f_over_r, dx);
                svfloat32_t fy_v = svmul_f32_z(valid, f_over_r, dy);
                svfloat32_t fz_v = svmul_f32_z(valid, f_over_r, dz);
                
                /* Accumulate i-forces in registers */
                fx_acc = svadd_f32_m(valid, fx_acc, fx_v);
                fy_acc = svadd_f32_m(valid, fy_acc, fy_v);
                fz_acc = svadd_f32_m(valid, fz_acc, fz_v);
                
                /* Scatter j-forces to memory via buffer */
                float fx_buf_out[vl], fy_buf_out[vl], fz_buf_out[vl];
                svst1_f32(pg, fx_buf_out, fx_v);
                svst1_f32(pg, fy_buf_out, fy_v);
                svst1_f32(pg, fz_buf_out, fz_v);
                svbool_t check = svptrue_b32();
                for (int m = 0; m < remaining && m < vl; m++) {
                    int j = j_atoms[k + m];
                    float rsq_j;
                    float dx_j = x[i*3+0] - x[j*3+0];
                    float dy_j = x[i*3+1] - x[j*3+1];
                    float dz_j = x[i*3+2] - x[j*3+2];
                    dx_j -= box[0] * roundf(dx_j / box[0]);
                    dy_j -= box[1] * roundf(dy_j / box[1]);
                    dz_j -= box[2] * roundf(dz_j / box[2]);
                    rsq_j = dx_j*dx_j + dy_j*dy_j + dz_j*dz_j;
                    if (rsq_j < cutoff_sq && rsq_j > 1e-8f) {
                        #pragma omp atomic
                        f[j*3+0] -= fx_buf_out[m];
                        #pragma omp atomic
                        f[j*3+1] -= fy_buf_out[m];
                        #pragma omp atomic
                        f[j*3+2] -= fz_buf_out[m];
                    }
                }
            }
        }
        
        /* Reduce i-forces using SVE horizontal reduction */
        fx_i = svaddv_f32(all_p, fx_acc);
        fy_i = svaddv_f32(all_p, fy_acc);
        fz_i = svaddv_f32(all_p, fz_acc);
        
        #pragma omp atomic
        f[i*3+0] += fx_i;
        #pragma omp atomic
        f[i*3+1] += fy_i;
        #pragma omp atomic
        f[i*3+2] += fz_i;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec + ts.tv_nsec * 1e-9) - t0;
}

/* ============ Main ============ */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <file.gro> [cutoff_nm] [nsteps]\n", argv[0]);
        printf("  file.gro  - GROMACS coordinate file\n");
        printf("  cutoff    - cutoff distance in nm (default: 1.0)\n");
        printf("  nsteps    - number of MD steps to simulate (default: 100)\n");
        return 1;
    }
    
    const char *gro_file = argv[1];
    float cutoff = (argc > 2) ? atof(argv[2]) : 1.0f;
    int nsteps = (argc > 3) ? atoi(argv[3]) : 100;
    
    printf("===== PTO-GROMACS 端到端 Benchmark =====\n");
    printf("File: %s\n", gro_file);
    printf("Cutoff: %.2f nm\n", cutoff);
    printf("Steps: %d\n", nsteps);
    printf("SVE vector length: %d bits (%d floats)\n", svcntb()*8, svcntw());
    printf("Threads: %d\n", omp_get_max_threads());
    printf("\n");
    
    /* Read GROMACS coordinate file */
    GroData *gro = read_gro(gro_file);
    if (!gro) return 1;
    printf("Atoms: %d\n", gro->natoms);
    printf("Box: %.3f x %.3f x %.3f nm\n", gro->box[0], gro->box[1], gro->box[2]);
    
    /* Build neighbor list */
    printf("\nBuilding Verlet neighbor list (cutoff=%.2f nm)...\n", cutoff);
    NeighborList *nl = build_neighbor_list(gro->x, gro->natoms, gro->box, cutoff);
    printf("Total pairs: %d (avg %.1f neighbors/atom)\n", 
           nl->total_pairs, 2.0f * nl->total_pairs / gro->natoms);
    
    float *f_scalar = malloc(gro->natoms * 3 * sizeof(float));
    float *f_pto = malloc(gro->natoms * 3 * sizeof(float));
    
    /* Warmup */
    compute_forces_scalar(gro->x, f_scalar, gro->natoms, gro->box, nl, cutoff);
    compute_forces_pto_sve(gro->x, f_pto, gro->natoms, gro->box, nl, cutoff);
    
    printf("\n--- Scalar baseline (GROMACS-style) ---\n");
    double total_scalar = 0;
    for (int s = 0; s < nsteps; s++) {
        total_scalar += compute_forces_scalar(gro->x, f_scalar, gro->natoms, gro->box, nl, cutoff);
    }
    printf("  Total: %.4f s | Per step: %.4f ms | %.1f M pairs/s\n",
           total_scalar, total_scalar/nsteps*1000,
           (double)nl->total_pairs * nsteps / total_scalar / 1e6);
    
    printf("\n--- PTO-SVE optimized (fused operators) ---\n");
    double total_pto = 0;
    for (int s = 0; s < nsteps; s++) {
        total_pto += compute_forces_pto_sve(gro->x, f_pto, gro->natoms, gro->box, nl, cutoff);
    }
    printf("  Total: %.4f s | Per step: %.4f ms | %.1f M pairs/s\n",
           total_pto, total_pto/nsteps*1000,
           (double)nl->total_pairs * nsteps / total_pto / 1e6);
    
    printf("\n===== 结果 =====\n");
    printf("Speedup: %.2fx\n", total_scalar / total_pto);
    printf("Scalar: %.4f ms/step | PTO-SVE: %.4f ms/step\n",
           total_scalar/nsteps*1000, total_pto/nsteps*1000);
    
    /* Force validation */
    float max_diff = 0;
    for (int i = 0; i < gro->natoms * 3; i++) {
        float d = fabsf(f_scalar[i] - f_pto[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("Force max diff: %.6e (should be small)\n", max_diff);
    
    /* Estimate ns/day equivalent (20fs timestep, just nonbonded portion) */
    double step_ns = 0.002;  /* 2fs */
    double ns_per_day_scalar = step_ns * nsteps / total_scalar * 86400;
    double ns_per_day_pto = step_ns * nsteps / total_pto * 86400;
    printf("\nEstimated nonbonded-only throughput:\n");
    printf("  Scalar: %.1f ns/day | PTO-SVE: %.1f ns/day\n", ns_per_day_scalar, ns_per_day_pto);
    
    free(f_scalar); free(f_pto);
    free_neighbor_list(nl);
    free_gro(gro);
    
    printf("\n===== 完成 =====\n");
    return 0;
}
