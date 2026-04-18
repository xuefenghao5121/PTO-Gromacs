/*
 * 鲲鹏930完整性能基准测试
 * PTO SVE优化 vs 标量基线
 */
#include "../gromacs_pto_arm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* 标量基线: 计算非键力 */
static void scalar_nonbonded(float *x, float *f, int n, float cutoff, float *charges) {
    float cutoff_sq = cutoff * cutoff;
    for (int i = 0; i < n; i++) {
        float fx=0, fy=0, fz=0;
        for (int j = i+1; j < n; j++) {
            float dx = x[i*3+0] - x[j*3+0];
            float dy = x[i*3+1] - x[j*3+1];
            float dz = x[i*3+2] - x[j*3+2];
            float rsq = dx*dx + dy*dy + dz*dz;
            if (rsq < cutoff_sq && rsq > 1e-6f) {
                /* LJ力 (简化参数) */
                float sigma = 0.3f, eps = 0.5f;
                float inv_rsq = 1.0f / rsq;
                float s2 = sigma*sigma * inv_rsq;
                float s6 = s2*s2*s2;
                float s12 = s6*s6;
                float f_lj = 24.0f * eps * (2.0f*s12 - s6) * inv_rsq;
                
                /* 库仑力 */
                float qq = charges[i] * charges[j];
                float r = sqrtf(rsq);
                float f_coul = qq / rsq;
                
                float f_over_r = f_lj + f_coul;
                fx += f_over_r * dx;
                fy += f_over_r * dy;
                fz += f_over_r * dz;
                f[j*3+0] -= f_over_r * dx;
                f[j*3+1] -= f_over_r * dy;
                f[j*3+2] -= f_over_r * dz;
            }
        }
        f[i*3+0] += fx;
        f[i*3+1] += fy;
        f[i*3+2] += fz;
    }
}

int main() {
    printf("===== 鲲鹏930 PTO-GROMACS 性能对比测试 =====\n\n");
    
    int sizes[] = {512, 1024, 2048, 4096};
    int num_sizes = 4;
    int repeats = 5;
    float cutoff = 1.2f;
    float box_size = 5.0f;
    
    printf("%-8s %-12s %-12s %-12s %-10s\n", "Atoms", "Scalar(ms)", "PTO-SVE(ms)", "Speedup", "Pairs");
    printf("------   ----------   ----------   ----------   ------\n");
    
    for (int si = 0; si < num_sizes; si++) {
        int n = sizes[si];
        float *x = malloc(n * 3 * sizeof(float));
        float *f_scalar = malloc(n * 3 * sizeof(float));
        float *f_pto = malloc(n * 3 * sizeof(float));
        float *charges = malloc(n * sizeof(float));
        
        /* 初始化: 保证原子间有最小距离 */
        srand(42);
        for (int i = 0; i < n; i++) {
            x[i*3+0] = (float)rand() / RAND_MAX * box_size;
            x[i*3+1] = (float)rand() / RAND_MAX * box_size;
            x[i*3+2] = (float)rand() / RAND_MAX * box_size;
            charges[i] = (i % 2 == 0 ? 1.0f : -1.0f) * 0.5f;
        }
        
        /* PTO setup */
        gmx_pto_config_t config;
        gmx_pto_config_init(&config);
        config.tile_size_atoms = 0;
        config.tile_size_cache_kb = 512;
        
        gmx_pto_nonbonded_context_t context;
        gmx_pto_create_tiling(n, x, &config, &context);
        gmx_pto_build_neighbor_pairs(&context, x, cutoff);
        context.params.cutoff_sq = cutoff * cutoff;
        context.params.epsilon_r = 1.0f;
        context.params.rf_kappa = 0.0f;
        context.params.charges = charges;
        
        gmx_pto_atom_data_t atom_data = {n, x, f_pto};
        
        /* 预热 */
        memset(f_scalar, 0, n*3*sizeof(float));
        memset(f_pto, 0, n*3*sizeof(float));
        scalar_nonbonded(x, f_scalar, n, cutoff, charges);
        gmx_pto_nonbonded_compute_fused(&context, &atom_data);
        
        /* 标量基准测试 */
        double t_scalar = 0;
        for (int r = 0; r < repeats; r++) {
            memset(f_scalar, 0, n*3*sizeof(float));
            double t0 = get_time();
            scalar_nonbonded(x, f_scalar, n, cutoff, charges);
            t_scalar += get_time() - t0;
        }
        t_scalar /= repeats;
        
        /* PTO-SVE基准测试 */
        double t_pto = 0;
        for (int r = 0; r < repeats; r++) {
            memset(f_pto, 0, n*3*sizeof(float));
            double t0 = get_time();
            gmx_pto_nonbonded_compute_fused(&context, &atom_data);
            t_pto += get_time() - t0;
        }
        t_pto /= repeats;
        
        double speedup = t_scalar / t_pto;
        long long pairs = (long long)n * (n-1) / 2;
        
        printf("%-8d %-12.3f %-12.3f %-12.2fx %-10lld\n", 
               n, t_scalar*1000, t_pto*1000, speedup, pairs);
        
        gmx_pto_destroy_tiling(&context);
        free(x); free(f_scalar); free(f_pto); free(charges);
    }
    
    printf("\n===== 测试完成 =====\n");
    return 0;
}
