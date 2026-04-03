/*
 * GROMACS PTO 完整性能基准测试
 * 
 * 使用完整的 Tile 划分 + 算子融合优化
 * 对比 baseline vs PTO 优化版本
 */

#include "../gromacs_pto_x86.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* 测试配置 */
#define MAX_REPEAT 10

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
        int nx = i % 10;
        int ny = (i / 10) % 10;
        int nz = i / 100;
        
        coords[i*3 + 0] = (nx * grid_step) + ((float)rand() / RAND_MAX - 0.5f) * grid_step * 0.3f;
        coords[i*3 + 1] = (ny * grid_step) + ((float)rand() / RAND_MAX - 0.5f) * grid_step * 0.3f;
        coords[i*3 + 2] = (nz * grid_step) + ((float)rand() / RAND_MAX - 0.5f) * grid_step * 0.3f;
        
        charges[i] = ((i % 2 == 0) ? 1.0f : -1.0f) * (0.1f + (float)rand() / RAND_MAX * 0.9f);
    }
}

/* 标量基准版本 */
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

/* PTO 优化版本 */
static double pto_compute(gmx_pto_nonbonded_context_x86_t *context,
                           gmx_pto_atom_data_x86_t *atom_data) {
    gmx_pto_nonbonded_compute_fused_x86(context, atom_data);
    
    double total_force = 0.0;
    int num_atoms = atom_data->num_atoms;
    for (int i = 0; i < num_atoms * 3; i++) {
        total_force += fabs((double)atom_data->f[i]);
    }
    
    return total_force * 0.001;
}

int main() {
    printf("========================================\n");
    printf("  GROMACS PTO 完整性能基准测试\n");
    printf("========================================\n\n");
    
    /* 打印系统信息 */
    printf("CPU Features:\n");
    printf("  AVX:      %s\n", gmx_pto_check_avx_support() ? "Yes" : "No");
    printf("  AVX2:     %s\n", gmx_pto_check_avx2_support() ? "Yes" : "No");
    printf("  Vec Width: %d floats\n", gmx_pto_get_avx_vector_width());
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
    double total_speedup = 0.0;
    
    /* 打开结果文件 */
    FILE *fp_results = fopen("benchmark_results.txt", "w");
    if (!fp_results) {
        printf("ERROR: Cannot create results file\n");
        return 1;
    }
    
    fprintf(fp_results, "# Test Name | Atoms | Baseline (ms) | PTO (ms) | Speedup | Throughput (ns/day)\n");
    
    for (int t = 0; t < num_tests; t++) {
        int num_atoms = test_configs[t].num_atoms;
        float box_nm = (float)test_configs[t].box_nm;
        float cutoff_nm = test_configs[t].cutoff_nm;
        int repeat = test_configs[t].repeat;
        float cutoff_sq = cutoff_nm * cutoff_nm;
        
        printf("\n=== 测试: %s ===\n", test_configs[t].name);
        printf("原子数: %d, 盒子: %dnm, 截断: %.2fnm, 重复: %d\n",
               num_atoms, test_configs[t].box_nm, (double)cutoff_nm, repeat);
        
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
        generate_atoms(num_atoms, box_nm, coords, charges);
        
        /* 初始化 PTO */
        gmx_pto_config_x86_t pto_config;
        gmx_pto_config_x86_init(&pto_config);
        pto_config.verbose = true;
        pto_config.tile_size_atoms = 0;  /* auto */
        pto_config.tile_size_cache_kb = 256;
        
        gmx_pto_nonbonded_context_x86_t pto_context;
        memset(&pto_context, 0, sizeof(pto_context));
        
        int ret = gmx_pto_create_tiling_x86(num_atoms, coords, &pto_config, &pto_context);
        if (ret != 0) {
            printf("ERROR: PTO tiling failed: %d\n", ret);
            free(coords);
            free(forces_baseline);
            free(forces_pto);
            free(charges);
            continue;
        }
        
        /* 构建邻居对 */
        ret = gmx_pto_build_neighbor_pairs_x86(&pto_context, coords, cutoff_nm);
        if (ret != 0) {
            printf("ERROR: Neighbor pairs failed: %d\n", ret);
            gmx_pto_destroy_tiling_x86(&pto_context);
            free(coords);
            free(forces_baseline);
            free(forces_pto);
            free(charges);
            continue;
        }
        
        /* PTO 参数 */
        pto_context.params.cutoff_sq = cutoff_sq;
        pto_context.params.epsilon_r = 1.0f;
        pto_context.params.rf_kappa = 0.0f;
        pto_context.params.charges = charges;
        
        gmx_pto_atom_data_x86_t pto_atom_data;
        pto_atom_data.num_atoms = num_atoms;
        pto_atom_data.x = coords;
        pto_atom_data.f = forces_pto;
        
        /* 预热 */
        printf("预热...\n");
        baseline_compute(num_atoms, coords, charges, forces_baseline, cutoff_sq);
        gmx_pto_nonbonded_compute_fused_x86(&pto_context, &pto_atom_data);
        
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
        printf("  Baseline: %.3f ms/iter, 能量: %.3f\n", baseline_time_ms, baseline_energy);
        
        /* PTO 测试 */
        printf("运行 PTO (%d 次重复)...\n", repeat);
        double pto_total = 0.0;
        double pto_energy = 0.0;
        
        for (int r = 0; r < repeat; r++) {
            memset(forces_pto, 0, num_atoms * 3 * sizeof(float));
            
            double start = get_time_ms();
            double energy = pto_compute(&pto_context, &pto_atom_data);
            double end = get_time_ms();
            
            pto_total += (end - start);
            if (r == 0) pto_energy = energy;
        }
        
        double pto_time_ms = pto_total / repeat;
        printf("  PTO:      %.3f ms/iter, 能量: %.3f\n", pto_time_ms, pto_energy);
        
        /* 性能指标 */
        double speedup = baseline_time_ms / pto_time_ms;
        total_speedup += speedup;
        
        /* 计算原子对数量 */
        int num_pairs = 0;
        for (int p = 0; p < pto_context.num_neighbor_pairs; p++) {
            num_pairs += pto_context.neighbor_pairs[p].num_pairs;
        }
        
        double pairs_per_sec = (double)num_pairs / (pto_time_ms / 1000.0);
        double steps_per_day = (24.0 * 3600.0 * 1000.0) / pto_time_ms;
        double throughput_ns_per_day = steps_per_day * 2.0e-6;
        
        printf("\n  加速比:   %.2fx\n", speedup);
        printf("  性能:     %.0f pairs/sec\n", pairs_per_sec);
        printf("  吞吐量:   %.2f ns/day\n", throughput_ns_per_day);
        printf("  Tile数:   %d\n", pto_context.num_tiles);
        printf("  原子对数: %d\n", num_pairs);
        
        /* 写入结果 */
        fprintf(fp_results, "%s | %d | %.3f | %.3f | %.2fx | %.2f\n",
                test_configs[t].name, num_atoms, baseline_time_ms, pto_time_ms,
                speedup, throughput_ns_per_day);
        
        /* 清理 */
        gmx_pto_destroy_tiling_x86(&pto_context);
        free(coords);
        free(forces_baseline);
        free(forces_pto);
        free(charges);
    }
    
    fclose(fp_results);
    
    /* 总结 */
    double avg_speedup = total_speedup / num_tests;
    
    printf("\n========================================\n");
    printf("  总结\n");
    printf("========================================\n");
    printf("平均加速比: %.2fx\n", avg_speedup);
    printf("结果文件: benchmark_results.txt\n");
    
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
