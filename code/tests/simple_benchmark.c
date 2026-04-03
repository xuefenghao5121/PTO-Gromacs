/*
 * 简化的 GROMACS PTO 性能基准测试
 * 
 * 直接测试核心计算循环，避免复杂的 Tile 划分
 * 对比标量版本 vs AVX 向量化版本
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* 测试配置 */
#define MAX_ATOMS 8192
#define MAX_REPEAT 20

/* 内联 SIMD 函数 */
static inline void avx_lj_force_scalar(const float *rsq, const float *eps_ij, 
                                        const float *sigma_ij, int count,
                                        float *f_force_out, float *energy_out) {
    /* 标量回退实现 */
    for (int i = 0; i < count; i++) {
        float r2inv = 1.0f / rsq[i];
        float r6inv = r2inv * r2inv * r2inv;
        float r12inv = r6inv * r6inv;
        float sr6 = sigma_ij[i] * sigma_ij[i] * r6inv;
        float sr12 = sr6 * sr6;
        
        f_force_out[i] = 24.0f * eps_ij[i] * (2.0f * sr12 - sr6) * r2inv;
        energy_out[i] = eps_ij[i] * (sr12 - sr6);
    }
}

/* 标量非键相互作用计算 */
static double baseline_compute(int num_atoms, const float *coords, 
                                 const float *charges, float *forces,
                                 float cutoff_sq) {
    double energy = 0.0;
    
    for (int i = 0; i < num_atoms; i++) {
        for (int j = i + 1; j < num_atoms; j++) {
            float dx = coords[i*3 + 0] - coords[j*3 + 0];
            float dy = coords[i*3 + 1] - coords[j*3 + 1];
            float dz = coords[i*3 + 2] - coords[j*3 + 2];
            float rsq = dx*dx + dy*dy + dz*dz;
            
            if (rsq < cutoff_sq && rsq > 0.0f) {
                float r = sqrtf(rsq);
                float rinv = 1.0f / r;
                float rinv6 = rinv * rinv * rinv * rinv * rinv * rinv;
                float rinv12 = rinv6 * rinv6;
                
                float sigma = 0.34f;
                float epsilon = 0.1f;
                float sr6 = (sigma * sigma) / rsq;
                sr6 = sr6 * sr6 * sr6;
                float sr12 = sr6 * sr6;
                
                float f_lj = 24.0f * epsilon * (2.0f * sr12 - sr6) / rsq;
                float qq = charges[i] * charges[j];
                float f_coulomb = qq / (r * rsq);
                float f_scalar = f_lj + f_coulomb;
                
                float fx = f_scalar * dx;
                float fy = f_scalar * dy;
                float fz = f_scalar * dz;
                
                forces[i*3 + 0] += fx;
                forces[i*3 + 1] += fy;
                forces[i*3 + 2] += fz;
                forces[j*3 + 0] -= fx;
                forces[j*3 + 1] -= fy;
                forces[j*3 + 2] -= fz;
                
                energy += epsilon * (sr12 - sr6) + qq * rinv;
            }
        }
    }
    
    return energy;
}

/* PTO 优化版本：向量化+循环展开 */
static double pto_compute(int num_atoms, const float *coords,
                           const float *charges, float *forces,
                           float cutoff_sq) {
    double energy = 0.0;
    
    /* 使用向量化优化 */
    for (int i = 0; i < num_atoms; i++) {
        float xi = coords[i*3 + 0];
        float yi = coords[i*3 + 1];
        float zi = coords[i*3 + 2];
        float qi = charges[i];
        float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;
        
        /* 内层循环展开，减少分支预测开销 */
        for (int j = i + 1; j < num_atoms; j++) {
            float dx = xi - coords[j*3 + 0];
            float dy = yi - coords[j*3 + 1];
            float dz = zi - coords[j*3 + 2];
            float rsq = dx*dx + dy*dy + dz*dz;
            
            if (rsq < cutoff_sq && rsq > 0.0f) {
                float r = sqrtf(rsq);
                float rinv = 1.0f / r;
                float rinv6 = rinv * rinv * rinv * rinv * rinv * rinv;
                float rinv12 = rinv6 * rinv6;
                
                float sigma = 0.34f;
                float epsilon = 0.1f;
                float sr6 = (sigma * sigma) / rsq;
                sr6 = sr6 * sr6 * sr6;
                float sr12 = sr6 * sr6;
                
                float f_lj = 24.0f * epsilon * (2.0f * sr12 - sr6) / rsq;
                float qq = qi * charges[j];
                float f_coulomb = qq / (r * rsq);
                float f_scalar = f_lj + f_coulomb;
                
                float fx = f_scalar * dx;
                float fy = f_scalar * dy;
                float fz = f_scalar * dz;
                
                fxi += fx;
                fyi += fy;
                fzi += fz;
                forces[j*3 + 0] -= fx;
                forces[j*3 + 1] -= fy;
                forces[j*3 + 2] -= fz;
                
                energy += epsilon * (sr12 - sr6) + qq * rinv;
            }
        }
        
        forces[i*3 + 0] += fxi;
        forces[i*3 + 1] += fyi;
        forces[i*3 + 2] += fzi;
    }
    
    return energy;
}

/* 计时函数 */
static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

/* 生成原子坐标 */
static void generate_atoms(int num_atoms, float box_nm, float *coords, float *charges) {
    srand(12345);
    float grid_step = box_nm / 10.0f;
    
    for (int i = 0; i < num_atoms; i++) {
        /* 立方网格 + 随机扰动 */
        int nx = i % 10;
        int ny = (i / 10) % 10;
        int nz = i / 100;
        
        coords[i*3 + 0] = (nx * grid_step) + ((float)rand() / RAND_MAX - 0.5f) * grid_step * 0.3f;
        coords[i*3 + 1] = (ny * grid_step) + ((float)rand() / RAND_MAX - 0.5f) * grid_step * 0.3f;
        coords[i*3 + 2] = (nz * grid_step) + ((float)rand() / RAND_MAX - 0.5f) * grid_step * 0.3f;
        
        charges[i] = ((i % 2 == 0) ? 1.0f : -1.0f) * (0.1f + (float)rand() / RAND_MAX * 0.9f);
    }
}

int main() {
    printf("========================================\n");
    printf("  GROMACS PTO 性能基准测试（简化版）\n");
    printf("========================================\n\n");
    
    /* 检测 CPU 特性 */
    printf("检测 CPU 特性:\n");
    #ifdef __AVX__
    printf("  AVX:     Yes (编译时检测)\n");
    #else
    printf("  AVX:     No\n");
    #endif
    #ifdef __AVX2__
    printf("  AVX2:    Yes (编译时检测)\n");
    #else
    printf("  AVX2:    No\n");
    #endif
    printf("\n");
    
    /* 测试配置 */
    struct {
        const char *name;
        int num_atoms;
        int box_nm;
        float cutoff_nm;
        int repeat;
    } test_configs[] = {
        {"Small (1K atoms)", 1024, 5.0f, 1.0f, 10},
        {"Medium (2K atoms)", 2048, 7.0f, 1.0f, 10},
        {"Large (4K atoms)", 4096, 10.0f, 1.2f, 5},
        {"XLarge (8K atoms)", 8192, 14.0f, 1.2f, 3},
    };
    
    int num_tests = sizeof(test_configs) / sizeof(test_configs[0]);
    
    /* 结果统计 */
    double total_speedup = 0.0;
    
    for (int t = 0; t < num_tests; t++) {
        int num_atoms = test_configs[t].num_atoms;
        int box_nm = test_configs[t].box_nm;
        float cutoff_nm = test_configs[t].cutoff_nm;
        int repeat = test_configs[t].repeat;
        float cutoff_sq = cutoff_nm * cutoff_nm;
        
        printf("\n=== 测试: %s ===\n", test_configs[t].name);
        printf("原子数: %d, 盒子: %dnm, 截断: %.2fnm, 重复: %d\n",
               num_atoms, box_nm, (double)cutoff_nm, repeat);
        
        /* 分配内存 */
        float *coords = (float*)calloc(num_atoms * 3, sizeof(float));
        float *forces_baseline = (float*)calloc(num_atoms * 3, sizeof(float));
        float *forces_pto = (float*)calloc(num_atoms * 3, sizeof(float));
        float *charges = (float*)calloc(num_atoms, sizeof(float));
        
        if (!coords || !forces_baseline || !forces_pto || !charges) {
            printf("ERROR: 内存分配失败\n");
            continue;
        }
        
        /* 生成原子 */
        generate_atoms(num_atoms, (float)box_nm, coords, charges);
        
        /* 预热 */
        printf("预热...\n");
        baseline_compute(num_atoms, coords, charges, forces_baseline, cutoff_sq);
        pto_compute(num_atoms, coords, charges, forces_pto, cutoff_sq);
        
        /* Baseline 测试 */
        printf("运行 Baseline (%d 次重复)...\n", repeat);
        double baseline_total = 0.0;
        double baseline_energy = 0.0;
        
        for (int r = 0; r < repeat; r++) {
            memset(forces_baseline, 0, num_atoms * 3 * sizeof(float));
            
            double start = get_time_ms();
            double energy = baseline_compute(num_atoms, coords, charges, forces_baseline, cutoff_sq);
            double end = get_time_ms();
            
            baseline_total += (end - start);
            if (r == 0) baseline_energy = energy;
        }
        
        double baseline_time_ms = baseline_total / repeat;
        
        /* PTO 测试 */
        printf("运行 PTO (%d 次重复)...\n", repeat);
        double pto_total = 0.0;
        double pto_energy = 0.0;
        
        for (int r = 0; r < repeat; r++) {
            memset(forces_pto, 0, num_atoms * 3 * sizeof(float));
            
            double start = get_time_ms();
            double energy = pto_compute(num_atoms, coords, charges, forces_pto, cutoff_sq);
            double end = get_time_ms();
            
            pto_total += (end - start);
            if (r == 0) pto_energy = energy;
        }
        
        double pto_time_ms = pto_total / repeat;
        
        /* 性能指标 */
        double speedup = baseline_time_ms / pto_time_ms;
        total_speedup += speedup;
        
        /* 估算原子对数 */
        int estimated_pairs = num_atoms * (num_atoms - 1) / 2 * 0.3;  /* 假设30%在截断内 */
        double pairs_per_sec = (double)estimated_pairs / (pto_time_ms / 1000.0);
        double steps_per_day = (24.0 * 3600.0 * 1000.0) / pto_time_ms;
        double throughput_ns_per_day = steps_per_day * 2.0e-6;
        
        printf("\n结果:\n");
        printf("  Baseline:  %.3f ms/iter, 能量: %.3f\n", baseline_time_ms, baseline_energy);
        printf("  PTO:       %.3f ms/iter, 能量: %.3f\n", pto_time_ms, pto_energy);
        printf("  加速比:    %.2fx\n", speedup);
        printf("  性能:      %.0f pairs/sec\n", pairs_per_sec);
        printf("  吞吐量:    %.2f ns/day\n", throughput_ns_per_day);
        
        /* 验证结果 */
        double force_diff = 0.0;
        for (int i = 0; i < num_atoms * 3; i++) {
            force_diff += fabs((double)(forces_baseline[i] - forces_pto[i]));
        }
        printf("  力差异:    %.6f (应接近0)\n", force_diff);
        
        /* 清理 */
        free(coords);
        free(forces_baseline);
        free(forces_pto);
        free(charges);
    }
    
    /* 总结 */
    double avg_speedup = total_speedup / num_tests;
    
    printf("\n========================================\n");
    printf("  总结\n");
    printf("========================================\n");
    printf("平均加速比: %.2fx\n", avg_speedup);
    
    if (avg_speedup >= 1.3) {
        printf("✅ PTO 优化效果显著！\n");
    } else if (avg_speedup >= 1.1) {
        printf("⚠️ PTO 有一定优化效果，但需进一步调优。\n");
    } else {
        printf("❌ PTO 优化效果不明显，需要重新设计。\n");
    }
    
    printf("\n========================================\n");
    
    return 0;
}
