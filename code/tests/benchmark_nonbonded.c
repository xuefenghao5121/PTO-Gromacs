/*
 * 性能基准测试：非键相互作用PTO优化
 * 
 * 测量完整计算流程的耗时
 * 对比标量版本和PTO优化版本的性能
 */

#include "../gromacs_pto_arm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* 测试大小 */
#define TEST_NUM_ATOMS 1024
#define TEST_REPEAT 10

/* 计时函数 */
static double get_time_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main() {
    printf("===== GROMACS PTO Nonbonded Performance Benchmark =====\n\n");
    
    /* 分配内存 */
    int num_atoms = TEST_NUM_ATOMS;
    float *coords = (float*)malloc(num_atoms * 3 * sizeof(float));
    float *forces = (float*)malloc(num_atoms * 3 * sizeof(float));
    float *charges = (float*)malloc(num_atoms * sizeof(float));
    
    if (!coords || !forces || !charges) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    /* 随机初始化数据 */
    printf("Initializing %d random atoms in 10nm box...\n", num_atoms);
    srand(12345);
    for (int i = 0; i < num_atoms; i++) {
        coords[i*3 + 0] = (float)rand() / RAND_MAX * 10.0f;
        coords[i*3 + 1] = (float)rand() / RAND_MAX * 10.0f;
        coords[i*3 + 2] = (float)rand() / RAND_MAX * 10.0f;
        forces[i*3 + 0] = 0.0f;
        forces[i*3 + 1] = 0.0f;
        forces[i*3 + 2] = 0.0f;
        charges[i] = (float)(i % 2 == 0 ? 1 : -1) * 0.5f;
    }
    
    /* 初始化PTO */
    gmx_pto_config_t config;
    gmx_pto_config_init(&config);
    config.verbose = true;
    config.tile_size_atoms = 0;  /* auto */
    config.tile_size_cache_kb = 512;
    
    int auto_tile = gmx_pto_auto_tile_size(num_atoms, config.tile_size_cache_kb);
    printf("Auto tile size: %d atoms\n", auto_tile);
    
    gmx_pto_nonbonded_context_t context;
    int ret = gmx_pto_create_tiling(num_atoms, coords, &config, &context);
    if (ret != 0) {
        printf("create_tiling failed: %d\n", ret);
        return 1;
    }
    
    /* 构建邻居对 */
    float cutoff = 1.0f;
    ret = gmx_pto_build_neighbor_pairs(&context, coords, cutoff);
    if (ret != 0) {
        printf("build_neighbor_pairs failed: %d\n", ret);
        gmx_pto_destroy_tiling(&context);
        return 1;
    }
    
    /* 设置参数 */
    context.params.cutoff_sq = cutoff * cutoff;
    context.params.epsilon_r = 1.0f;
    context.params.rf_kappa = 0.0f;
    context.params.charges = charges;
    
    gmx_pto_atom_data_t atom_data;
    atom_data.num_atoms = num_atoms;
    atom_data.x = coords;
    atom_data.f = forces;
    
    /* 预热 */
    printf("\nWarming up...\n");
    gmx_pto_nonbonded_compute_fused(&context, &atom_data);
    
    /* 性能测试 */
    printf("\nBenchmarking: %d repeats...\n", TEST_REPEAT);
    
    double start = get_time_seconds();
    
    for (int i = 0; i < TEST_REPEAT; i++) {
        /* 清零力 */
        for (int a = 0; a < num_atoms * 3; a++) {
            forces[a] = 0.0f;
        }
        
        gmx_pto_nonbonded_compute_fused(&context, &atom_data);
    }
    
    double end = get_time_seconds();
    double total_time = end - start;
    double time_per_iteration = total_time / TEST_REPEAT;
    
    /* 计算相互作用对数 */
    int total_pairs = 0;
    for (int p = 0; p < context.num_neighbor_pairs; p++) {
        total_pairs += context.neighbor_pairs[p].num_pairs;
    }
    
    /* 统计结果 */
    printf("\n======== Benchmark Results ========\n");
    printf("Total atoms:            %d\n", num_atoms);
    printf("Number of tiles:         %d\n", context.num_tiles);
    printf("Number of neighbor pairs: %d\n", context.num_neighbor_pairs);
    printf("Total atom pairs:        %d\n", total_pairs);
    printf("Repeats:                 %d\n", TEST_REPEAT);
    printf("Total time:              %.3f seconds\n", total_time);
    printf("Time per iteration:      %.3f ms\n", time_per_iteration * 1000);
    printf("Performance:             %.0f pairs/second\n",
           (double)total_pairs * TEST_REPEAT / total_time);
    printf("Performance per atom:    %.3f us/atom\n",
           time_per_iteration * 1e6 / num_atoms);
    
    /* SVE信息 */
    if (config.enable_sve) {
        printf("SVE vector length:       %d bits (%d floats)\n",
               gmx_pto_get_sve_vector_length_bits(),
               gmx_pto_get_sve_vector_length_floats());
    }
    
    /* SME信息 */
    if (context.sme_enabled) {
        printf("SME:                     Enabled and in use\n");
    } else {
        printf("SME:                     %s\n",
               config.enable_sme ? "Not available" : "Disabled");
    }
    
    printf("===================================\n");
    
    /* 计算总力大小，检查计算结果 */
    double total_force = 0.0;
    for (int i = 0; i < num_atoms * 3; i++) {
        total_force += fabs((double)forces[i]);
    }
    printf("Check: total force magnitude = %.3f (should not be zero)\n", total_force);
    
    /* 清理 */
    gmx_pto_destroy_tiling(&context);
    free(coords);
    free(forces);
    free(charges);
    
    printf("\nBenchmark completed.\n");
    return 0;
}
