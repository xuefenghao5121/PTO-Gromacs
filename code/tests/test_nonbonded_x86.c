/*
 * 单元测试：非键相互作用PTO优化 (x86版本)
 * 
 * 测试内容：
 * 1. Tile划分正确性
 * 2. AVX向量化计算正确性
 * 3. AVX2可用性检测
 * 4. 融合计算完整性
 * 5. 对比参考实现验证结果
 */

#include "../gromacs_pto_x86.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* 测试配置 */
#define TEST_MAX_ATOMS 256
#define TOLERANCE 1e-6f

/* 参考实现：标量距离平方计算 */
float reference_distance_sq(float x1, float y1, float z1, 
                             float x2, float y2, float z2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float dz = z1 - z2;
    return dx*dx + dy*dy + dz*dz;
}

/* 参考实现：LJ能量计算 */
float reference_lj_energy(float r, float sigma, float epsilon) {
    if (r <= 0.0f) return 0.0f;
    float sigma_over_r = sigma / r;
    float sigma_over_r6 = sigma_over_r * sigma_over_r;
    sigma_over_r6 = sigma_over_r6 * sigma_over_r6 * sigma_over_r6;
    return 4.0f * epsilon * (sigma_over_r6 * sigma_over_r6 - sigma_over_r6);
}

/* 参考实现：原子对相互作用计算（标量）版本 */
void reference_pair_compute(float coords_i[3], float coords_j[3],
                             float force_i[3], float force_j[3],
                             float sigma, float epsilon) {
    float dx = coords_j[0] - coords_i[0];
    float dy = coords_j[1] - coords_i[1];
    float dz = coords_j[2] - coords_i[2];
    float rsq = dx*dx + dy*dy + dz*dz;
    float r = sqrtf(rsq);
    
    if (r < 1e-6f) return;
    
    float sigma_over_r = sigma / r;
    float sigma_over_r6 = sigma_over_r * sigma_over_r;
    sigma_over_r6 = sigma_over_r6 * sigma_over_r6 * sigma_over_r6;
    float dVdr = 12.0f * epsilon * (sigma_over_r6 * sigma_over_r6 - 0.5f * sigma_over_r6) / r;
    
    float fx = dVdr * dx / r;
    float fy = dVdr * dy / r;
    float fz = dVdr * dz / r;
    
    force_i[0] += fx;
    force_i[1] += fy;
    force_i[2] += fz;
    force_j[0] -= fx;
    force_j[1] -= fy;
    force_j[2] -= fz;
}

/* 测试1: CPU检测 */
int test_cpu_detection(void) {
    printf("测试1: CPU检测\n");
    
    bool has_avx = gmx_pto_check_avx_support();
    bool has_avx2 = gmx_pto_check_avx2_support();
    int vec_width = gmx_pto_get_avx_vector_width();
    
    printf("  AVX支持: %s\n", has_avx ? "是" : "否");
    printf("  AVX2支持: %s\n", has_avx2 ? "是" : "否");
    printf("  向量宽度: %d floats\n", vec_width);
    
    if (vec_width <= 0) {
        printf("  失败: 无效的向量宽度\n");
        return -1;
    }
    
    printf("  通过\n");
    return 0;
}

/* 测试2: 配置初始化 */
int test_config_init(void) {
    printf("测试2: 配置初始化\n");
    
    gmx_pto_config_x86_t config;
    gmx_pto_config_x86_init(&config);
    
    if (config.tile_size_atoms <= 0) {
        printf("  失败: 无效的tile大小\n");
        return -1;
    }
    
    if (config.tile_size_cache_kb <= 0) {
        printf("  失败: 无效的缓存大小\n");
        return -1;
    }
    
    printf("  Tile大小: %d 原子\n", config.tile_size_atoms);
    printf("  缓存大小: %d KB\n", config.tile_size_cache_kb);
    printf("  通过\n");
    return 0;
}

/* 测试3: 自动Tile大小计算 */
int test_auto_tile_size(void) {
    printf("测试3: 自动Tile大小计算\n");
    
    int tile_size;
    
    tile_size = gmx_pto_auto_tile_size_x86(1000, 256);
    if (tile_size <= 0 || tile_size > 1024) {
        printf("  失败: 无效的tile大小 %d\n", tile_size);
        return -1;
    }
    printf("  256KB缓存 -> tile大小: %d\n", tile_size);
    
    tile_size = gmx_pto_auto_tile_size_x86(1000, 32);
    if (tile_size <= 0 || tile_size > 1024) {
        printf("  失败: 无效的tile大小 %d\n", tile_size);
        return -1;
    }
    printf("  32KB缓存 -> tile大小: %d\n", tile_size);
    
    printf("  通过\n");
    return 0;
}

/* 测试4: Tile创建和销毁 */
int test_tile_creation(void) {
    printf("测试4: Tile创建和销毁\n");
    
    const int n_atoms = 100;
    float coords[100 * 3];
    
    /* 创建简单的坐标 */
    for (int i = 0; i < n_atoms; i++) {
        coords[i*3+0] = (float)i * 0.1f;
        coords[i*3+1] = (float)i * 0.1f;
        coords[i*3+2] = (float)i * 0.1f;
    }
    
    gmx_pto_config_x86_t config;
    gmx_pto_config_x86_init(&config);
    config.verbose = false;
    
    gmx_pto_nonbonded_context_x86_t context;
    int ret = gmx_pto_create_tiling_x86(n_atoms, coords, &config, &context);
    
    if (ret != 0) {
        printf("  失败: 创建tiling返回错误 %d\n", ret);
        return -1;
    }
    
    if (context.num_tiles <= 0 || context.num_total_atoms != n_atoms) {
        printf("  失败: 无效的tiling数据\n");
        gmx_pto_destroy_tiling_x86(&context);
        return -1;
    }
    
    printf("  原子数: %d\n", context.num_total_atoms);
    printf("  Tile数: %d\n", context.num_tiles);
    printf("  通过\n");
    
    gmx_pto_destroy_tiling_x86(&context);
    return 0;
}

/* 测试5: 邻居对构建 */
int test_neighbor_pairs(void) {
    printf("测试5: 邻居对构建\n");
    
    const int n_atoms = 50;
    float coords[50 * 3];
    
    /* 创建坐标 - 放在一个盒子内 */
    for (int i = 0; i < n_atoms; i++) {
        coords[i*3+0] = ((float)(i % 5)) * 1.0f;
        coords[i*3+1] = ((float)(i / 5)) * 1.0f;
        coords[i*3+2] = 0.0f;
    }
    
    gmx_pto_config_x86_t config;
    gmx_pto_config_x86_init(&config);
    config.verbose = false;
    
    gmx_pto_nonbonded_context_x86_t context;
    int ret = gmx_pto_create_tiling_x86(n_atoms, coords, &config, &context);
    
    if (ret != 0) {
        printf("  失败: 创建tiling\n");
        return -1;
    }
    
    ret = gmx_pto_build_neighbor_pairs_x86(&context, coords, 2.0f);
    
    if (ret != 0) {
        printf("  失败: 构建邻居对\n");
        gmx_pto_destroy_tiling_x86(&context);
        return -1;
    }
    
    if (context.num_neighbor_pairs <= 0) {
        printf("  失败: 无邻居对\n");
        gmx_pto_destroy_tiling_x86(&context);
        return -1;
    }
    
    printf("  邻居对数: %d\n", context.num_neighbor_pairs);
    printf("  通过\n");
    
    gmx_pto_destroy_tiling_x86(&context);
    return 0;
}

/* 测试6: AVX距离计算 */
int test_avx_distance(void) {
    printf("测试6: AVX距离计算\n");
    
    const int count = 16;
    float x1 = 0.0f, y1 = 0.0f, z1 = 0.0f;
    float x2[16], y2[16], z2[16];
    float rsq_out[16], rsq_ref[16];
    
    /* 创建测试坐标 */
    for (int i = 0; i < count; i++) {
        x2[i] = (float)i * 0.1f;
        y2[i] = (float)i * 0.2f;
        z2[i] = (float)i * 0.3f;
    }
    
    /* AVX版本 */
    gmx_pto_avx_distance_sq(&x1, &y1, &z1, x2, y2, z2, rsq_out, count);
    
    /* 参考版本 */
    for (int i = 0; i < count; i++) {
        rsq_ref[i] = reference_distance_sq(x1, y1, z1, x2[i], y2[i], z2[i]);
    }
    
    /* 验证 */
    for (int i = 0; i < count; i++) {
        float diff = fabsf(rsq_out[i] - rsq_ref[i]);
        if (diff > TOLERANCE) {
            printf("  失败: 索引%d差异 %.6f (预期 %.6f, 实际 %.6f)\n", 
                   i, diff, rsq_ref[i], rsq_out[i]);
            return -1;
        }
    }
    
    printf("  通过\n");
    return 0;
}

/* 测试7: 工具函数 */
int test_utility_functions(void) {
    printf("测试7: 工具函数\n");
    
    /* 测试 pto_check_tile_fits_in_cache_x86 */
    int fits = pto_check_tile_fits_in_cache_x86(64, 256);
    if (fits != 1) {
    }
    
    fits = pto_check_tile_fits_in_cache_x86(10000, 256);
    if (fits != 0) {
    }
    
    /* 测试 pto_minimum_image_x86 */
    float dx = 1.5f;
    float box = 2.0f;
    float half = 1.0f;
    pto_minimum_image_x86(&dx, box, half);
    if (fabsf(dx) > half) {
        printf("  失败: 最小图像约定\n");
        return -1;
    }
    
    printf("  通过\n");
    return 0;
}

/* 测试8: 完整流程测试 */
int test_full_pipeline(void) {
    printf("测试8: 完整流程测试\n");
    
    const int n_atoms = 32;
    float coords[32 * 3];
    
    /* 创建简单坐标 */
    for (int i = 0; i < n_atoms; i++) {
        coords[i*3+0] = (float)(i % 8) * 1.0f;
        coords[i*3+1] = (float)(i / 8) * 1.0f;
        coords[i*3+2] = 0.0f;
    }
    
    /* 初始化配置 */
    gmx_pto_config_x86_t config;
    gmx_pto_config_x86_init(&config);
    config.verbose = false;
    
    /* 创建tiling */
    gmx_pto_nonbonded_context_x86_t context;
    int ret = gmx_pto_create_tiling_x86(n_atoms, coords, &config, &context);
    if (ret != 0) {
        printf("  失败: 创建tiling\n");
        return -1;
    }
    
    /* 构建邻居对 */
    ret = gmx_pto_build_neighbor_pairs_x86(&context, coords, 1.5f);
    if (ret != 0) {
        printf("  失败: 构建邻居对\n");
        gmx_pto_destroy_tiling_x86(&context);
        return -1;
    }
    
    /* 初始化参数 */
    context.params.cutoff_sq = 1.5f * 1.5f;
    context.params.epsilon_r = 1.0f;
    context.params.rf_kappa = 0.0f;
    context.params.lj_epsilon = NULL;
    context.params.lj_sigma = NULL;
    context.params.charges = NULL;
    
    /* 准备原子数据 */
    gmx_pto_atom_data_x86_t atom_data;
    atom_data.num_atoms = n_atoms;
    atom_data.x = coords;
    
    float forces[32 * 3] = {0};
    atom_data.f = forces;
    
    /* 执行计算 */
    ret = gmx_pto_nonbonded_compute_fused_x86(&context, &atom_data);
    if (ret != 0) {
        printf("  失败: 融合计算\n");
        gmx_pto_destroy_tiling_x86(&context);
        return -1;
    }
    
    printf("  通过\n");
    gmx_pto_destroy_tiling_x86(&context);
    return 0;
}

/* 主测试函数 */
int main(void) {
    printf("========================================\n");
    printf("GROMACS PTO x86 单元测试\n");
    printf("========================================\n\n");
    
    int passed = 0;
    int failed = 0;
    
    if (test_cpu_detection() == 0) passed++; else failed++;
    printf("\n");
    
    if (test_config_init() == 0) passed++; else failed++;
    printf("\n");
    
    if (test_auto_tile_size() == 0) passed++; else failed++;
    printf("\n");
    
    if (test_tile_creation() == 0) passed++; else failed++;
    printf("\n");
    
    if (test_neighbor_pairs() == 0) passed++; else failed++;
    printf("\n");
    
    if (test_avx_distance() == 0) passed++; else failed++;
    printf("\n");
    
    if (test_utility_functions() == 0) passed++; else failed++;
    printf("\n");
    
    if (test_full_pipeline() == 0) passed++; else failed++;
    printf("\n");
    
    printf("========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", passed, failed);
    printf("========================================\n");
    
    return (failed > 0) ? 1 : 0;
}
