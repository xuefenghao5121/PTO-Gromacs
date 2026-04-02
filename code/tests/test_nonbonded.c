/*
 * 单元测试：非键相互作用PTO优化
 * 
 * 测试内容：
 * 1. Tile划分正确性
 * 2. SVE向量化计算正确性
 * 3. SME可用性检测
 * 4. 融合计算完整性
 * 5. 对比参考实现验证结果
 */

#include "../gromacs_pto_arm.h"
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

/* 参考实现：标量LJ力计算 */
void reference_lj_force(float rsq, float eps_ij, float sigma_ij,
                         float *f_force_out, float *energy_out) {
    float sigma_sq = sigma_ij * sigma_ij;
    float inv_rsq = 1.0f / rsq;
    float sig_inv_rsq = sigma_sq * inv_rsq;
    float t6 = sig_inv_rsq * sig_inv_rsq * sig_inv_rsq;
    float t12 = t6 * t6;
    
    *energy_out = 4.0f * eps_ij * (t12 - t6);
    float term = 2.0f * t12 - t6;
    *f_force_out = 24.0f * eps_ij * term * inv_rsq;
}

/* 测试1: 基础配置初始化 */
static int test_config_init() {
    printf("Test 1: Configuration initialization... ");
    gmx_pto_config_t config;
    gmx_pto_config_init(&config);
    
    if (config.tile_size_atoms != 64) return 1;
    if (config.enable_sve != true) return 1;
    if (config.enable_sme != true) return 1;
    if (config.enable_fusion != true) return 1;
    
    printf("OK\n");
    return 0;
}

/* 测试2: 自动Tile大小计算 */
static int test_auto_tile_size() {
    printf("Test 2: Auto tile size calculation... ");
    
    int size1 = gmx_pto_auto_tile_size(1000, 512);
    if (size1 < 16 || size1 > 1024) {
        printf("FAIL: size1 out of range %d\n", size1);
        return 1;
    }
    
    int size2 = gmx_pto_auto_tile_size(10000, 1024);
    if (size2 < 16 || size2 > 1024) {
        printf("FAIL: size2 out of range %d\n", size2);
        return 1;
    }
    
    /* 应该是2的幂 */
    if ((size1 & (size1 - 1)) != 0) {
        printf("FAIL: not power of two %d\n", size1);
        return 1;
    }
    
    printf("OK (size1=%d, size2=%d)\n", size1, size2);
    return 0;
}

/* 测试3: SVE向量长度查询 */
static int test_sve_length() {
    printf("Test 3: SVE vector length query... ");
    
    if (!svcntw()) {
        printf("SKIP (SVE not available on this host)\n");
        return 0;
    }
    
    int bits = gmx_pto_get_sve_vector_length_bits();
    int floats = gmx_pto_get_sve_vector_length_floats();
    
    if (bits != floats * 32) {
        printf("FAIL: bits=%d != floats*32=%d\n", bits, floats*32);
        return 1;
    }
    
    if (bits < 128 || bits > 2048) {
        printf("FAIL: bits out of range %d\n", bits);
        return 1;
    }
    
    printf("OK (%d bits, %d floats)\n", bits, floats);
    return 0;
}

/* 测试4: SME可用性检测 */
static int test_sme_available() {
    printf("Test 4: SME availability check... ");
    
    bool available = gmx_pto_sme_is_available();
    
    if (available) {
        printf("OK (SME available on this host)\n");
    } else {
        printf("OK (SME not available on this host - expected on non-ARMv9)\n");
    }
    
    return 0;
}

/* 测试5: Tile划分创建 */
static int test_create_tiling() {
    printf("Test 5: Tile creation... ");
    
    const int num_atoms = 100;
    float coords[num_atoms * 3];
    
    /* 随机生成坐标 */
    for (int i = 0; i < num_atoms * 3; i++) {
        coords[i] = (float)rand() / RAND_MAX * 10.0f;
    }
    
    gmx_pto_config_t config;
    gmx_pto_config_init(&config);
    config.tile_size_atoms = 32;
    config.verbose = false;
    
    gmx_pto_nonbonded_context_t context;
    int ret = gmx_pto_create_tiling(num_atoms, coords, &config, &context);
    
    if (ret != 0) {
        printf("FAIL: create_tiling returned %d\n", ret);
        return 1;
    }
    
    if (context.num_tiles != 4) {  /* 100 atoms / 32 = 4 tiles */
        printf("FAIL: wrong number of tiles %d != 4\n", context.num_tiles);
        gmx_pto_destroy_tiling(&context);
        return 1;
    }
    
    int total_atoms = 0;
    for (int t = 0; t < context.num_tiles; t++) {
        total_atoms += context.tiles[t].num_atoms;
        if (context.tiles[t].atom_indices == NULL) {
            printf("FAIL: tile %d has no atom indices\n", t);
            gmx_pto_destroy_tiling(&context);
            return 1;
        }
    }
    
    if (total_atoms != num_atoms) {
        printf("FAIL: total atoms %d != %d\n", total_atoms, num_atoms);
        gmx_pto_destroy_tiling(&context);
        return 1;
    }
    
    gmx_pto_destroy_tiling(&context);
    printf("OK (%d tiles created)\n", context.num_tiles);
    return 0;
}

/* 测试6: 邻居对构建 */
static int test_build_neighbor_pairs() {
    printf("Test 6: Neighbor pair building... ");
    
    const int num_atoms = 64;
    float coords[num_atoms * 3];
    
    /* 网格排列原子 */
    int n = 4;
    float spacing = 1.0f;
    int idx = 0;
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            for (int z = 0; z < n; z++) {
                coords[idx*3 + 0] = x * spacing;
                coords[idx*3 + 1] = y * spacing;
                coords[idx*3 + 2] = z * spacing;
                idx++;
            }
        }
    }
    
    gmx_pto_config_t config;
    gmx_pto_config_init(&config);
    config.tile_size_atoms = 32;
    
    gmx_pto_nonbonded_context_t context;
    int ret = gmx_pto_create_tiling(num_atoms, coords, &config, &context);
    if (ret != 0) {
        printf("FAIL: create_tiling %d\n", ret);
        return 1;
    }
    
    ret = gmx_pto_build_neighbor_pairs(&context, coords, 1.5f);
    if (ret != 0) {
        printf("FAIL: build_neighbor_pairs %d\n", ret);
        gmx_pto_destroy_tiling(&context);
        return 1;
    }
    
    if (context.num_neighbor_pairs == 0) {
        printf("FAIL: no neighbor pairs found\n");
        gmx_pto_destroy_tiling(&context);
        return 1;
    }
    
    /* 每个tile应该找到自己 */
    bool found_self = false;
    for (int i = 0; i < context.num_neighbor_pairs; i++) {
        if (context.neighbor_pairs[i].tile_i == context.neighbor_pairs[i].tile_j) {
            found_self = true;
            break;
        }
    }
    
    if (!found_self) {
        printf("FAIL: no tile pairs with i=j\n");
        gmx_pto_destroy_tiling(&context);
        return 1;
    }
    
    printf("OK (%d pairs built)\n", context.num_neighbor_pairs);
    gmx_pto_destroy_tiling(&context);
    return 0;
}

/* 测试7: SVE距离平方计算对比参考 */
static int test_sve_distance_sq() {
    printf("Test 7: SVE distance squared vs reference... ");
    
    if (!svcntw()) {
        printf("SKIP (SVE not available)\n");
        return 0;
    }
    
    int vl = gmx_pto_get_sve_vector_length_floats();
    float x1[vl], y1[vl], z1[vl];
    float x2[vl], y2[vl], z2[vl];
    float ref_rsq[vl];
    
    /* 生成随机数据 */
    for (int i = 0; i < vl; i++) {
        x1[i] = (float)rand() / RAND_MAX;
        y1[i] = (float)rand() / RAND_MAX;
        z1[i] = (float)rand() / RAND_MAX;
        x2[i] = (float)rand() / RAND_MAX;
        y2[i] = (float)rand() / RAND_MAX;
        z2[i] = (float)rand() / RAND_MAX;
        
        ref_rsq[i] = reference_distance_sq(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i]);
    }
    
    /* 加载到SVE向量 */
    svbool_t p = svptrue_b32();
    svfloat32_t sv_x1 = svld1_f32(p, x1);
    svfloat32_t sv_y1 = svld1_f32(p, y1);
    svfloat32_t sv_z1 = svld1_f32(p, z1);
    svfloat32_t sv_x2 = svld1_f32(p, x2);
    svfloat32_t sv_y2 = svld1_f32(p, y2);
    svfloat32_t sv_z2 = svld1_f32(p, z2);
    
    svfloat32_t sv_rsq = gmx_pto_sve_distance_sq(sv_x1, sv_y1, sv_z1,
                                                  sv_x2, sv_y2, sv_z2);
    
    /* 存储回内存对比 */
    float result[vl];
    svst1_f32(p, result, sv_rsq);
    
    /* 对比每个元素 */
    for (int i = 0; i < vl; i++) {
        if (fabsf(result[i] - ref_rsq[i]) > TOLERANCE) {
            printf("FAIL: index %d result=%f ref=%f diff=%f\n",
                   i, result[i], ref_rsq[i], result[i] - ref_rsq[i]);
            return 1;
        }
    }
    
    printf("OK (%d elements match)\n", vl);
    return 0;
}

/* 测试8: SVE LJ力计算对比参考 */
static int test_sve_lj_force() {
    printf("Test 8: SVE LJ force vs reference... ");
    
    if (!svcntw()) {
        printf("SKIP (SVE not available)\n");
        return 0;
    }
    
    int vl = gmx_pto_get_sve_vector_length_floats();
    float rsq_arr[vl];
    float eps_arr[vl];
    float sigma_arr[vl];
    float ref_f[vl], ref_e[vl];
    
    for (int i = 0; i < vl; i++) {
        rsq_arr[i] = 0.1f + 0.8f * (float)i / vl;  /* 0.1 to 0.9 */
        eps_arr[i] = 0.5f;
        sigma_arr[i] = 0.3f;
        reference_lj_force(rsq_arr[i], eps_arr[i], sigma_arr[i], &ref_f[i], &ref_e[i]);
    }
    
    svbool_t p = svptrue_b32();
    svfloat32_t rsq = svld1_f32(p, rsq_arr);
    svfloat32_t eps = svld1_f32(p, eps_arr);
    svfloat32_t sigma = svld1_f32(p, sigma_arr);
    
    svfloat32_t f_sve, e_sve;
    gmx_pto_sve_lj_force(rsq, eps, sigma, &f_sve, &e_sve);
    
    float f_res[vl], e_res[vl];
    svst1_f32(p, f_res, f_sve);
    svst1_f32(p, e_res, e_sve);
    
    for (int i = 0; i < vl; i++) {
        if (fabsf(f_res[i] - ref_f[i]) > TOLERANCE * 10) {
            printf("FAIL: f mismatch at %d: %f vs %f\n", i, f_res[i], ref_f[i]);
            return 1;
        }
        if (fabsf(e_res[i] - ref_e[i]) > TOLERANCE * 10) {
            printf("FAIL: e mismatch at %d: %f vs %f\n", i, e_res[i], ref_e[i]);
            return 1;
        }
    }
    
    printf("OK (%d elements match)\n", vl);
    return 0;
}

/* 测试9: 完整融合计算流程 */
static int test_full_fused_computation() {
    printf("Test 9: Full fused computation... ");
    
    const int num_atoms = 64;
    float coords[num_atoms * 3];
    float forces[num_atoms * 3];
    float charges[num_atoms];
    
    /* 初始化 */
    for (int i = 0; i < num_atoms; i++) {
        coords[i*3 + 0] = (float)(i % 4);
        coords[i*3 + 1] = (float)((i / 4) % 4);
        coords[i*3 + 2] = (float)(i / 16);
        forces[i*3 + 0] = 0.0f;
        forces[i*3 + 1] = 0.0f;
        forces[i*3 + 2] = 0.0f;
        charges[i] = 0.1f * (i % 2 == 0 ? 1 : -1);
    }
    
    gmx_pto_config_t config;
    gmx_pto_config_init(&config);
    config.tile_size_atoms = 32;
    config.verbose = false;
    
    gmx_pto_nonbonded_context_t context;
    int ret = gmx_pto_create_tiling(num_atoms, coords, &config, &context);
    if (ret != 0) {
        printf("FAIL: create_tiling %d\n", ret);
        return 1;
    }
    
    ret = gmx_pto_build_neighbor_pairs(&context, coords, 1.5f);
    if (ret != 0) {
        printf("FAIL: build_neighbor_pairs %d\n", ret);
        gmx_pto_destroy_tiling(&context);
        return 1;
    }
    
    /* 设置参数 */
    context.params.cutoff_sq = 1.5f * 1.5f;
    context.params.epsilon_r = 1.0f;
    context.params.rf_kappa = 0.0f;
    context.params.charges = charges;
    
    gmx_pto_atom_data_t atom_data;
    atom_data.num_atoms = num_atoms;
    atom_data.x = coords;
    atom_data.f = forces;
    
    ret = gmx_pto_nonbonded_compute_fused(&context, &atom_data);
    if (ret != 0) {
        printf("FAIL: compute_fused %d\n", ret);
        gmx_pto_destroy_tiling(&context);
        return 1;
    }
    
    /* 检查所有Tile都计算完成 */
    for (int t = 0; t < context.num_tiles; t++) {
        if (!context.tiles[t].forces_computed) {
            printf("FAIL: tile %d not computed\n", t);
            gmx_pto_destroy_tiling(&context);
            return 1;
        }
    }
    
    /* 检查力不为零（至少一些相互作用被计算） */
    float total_force = 0.0f;
    for (int i = 0; i < num_atoms * 3; i++) {
        total_force += fabsf(forces[i]);
    }
    
    if (total_force < 1e-6f) {
        printf("FAIL: total force is zero %f, no interactions computed\n", total_force);
        gmx_pto_destroy_tiling(&context);
        return 1;
    }
    
    printf("OK (total force magnitude: %f)\n", total_force);
    gmx_pto_destroy_tiling(&context);
    return 0;
}

/* 主测试入口 */
int main() {
    printf("\n===== GROMACS PTO ARM SVE/SME Unit Tests =====\n\n");
    
    int failed = 0;
    
    failed += test_config_init();
    failed += test_auto_tile_size();
    failed += test_sve_length();
    failed += test_sme_available();
    failed += test_create_tiling();
    failed += test_build_neighbor_pairs();
    failed += test_sve_distance_sq();
    failed += test_sve_lj_force();
    failed += test_full_fused_computation();
    
    printf("\n===== Test Summary =====\n");
    if (failed == 0) {
        printf("All tests passed! ✓\n");
    } else {
        printf("%d tests FAILED! ✗\n", failed);
    }
    printf("========================\n\n");
    
    return failed;
}
