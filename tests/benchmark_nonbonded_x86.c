/**
 * @file benchmark_nonbonded_x86.c
 * @brief 非键相互作用SIMD优化基准测试
 */

#include "test_framework_x86.h"
#include <math.h>

/* 测试参数 */
#define PARTICLE_COUNT 4096
#define BOX_SIZE 10.0f
#define CUTOFF 1.2f
#define CUTOFF_SQ (CUTOFF * CUTOFF)
#define WARMUP_ITER 10
#define BENCH_ITER 100

/* LJ参数 */
#define C6  0.001f
#define C12 0.0001f

/* ========== Scalar实现 ========== */

static float compute_nb_energy_scalar(
    const float *x, const float *y, const float *z,
    const float *q, size_t n, float cutoff_sq
) {
    float energy = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            float dx = x[j] - x[i];
            float dy = y[j] - y[i];
            float dz = z[j] - z[i];
            float r2 = dx*dx + dy*dy + dz*dz;
            
            if (r2 < cutoff_sq) {
                /* LJ能量 */
                energy += compute_lj_energy_scalar(r2, C6, C12);
                
                /* 库仑能量 */
                float r = sqrtf(r2);
                energy += compute_coulomb_energy_scalar(q[i], q[j], r);
            }
        }
    }
    
    return energy;
}

/* ========== SSE2实现 ========== */

#ifdef __SSE2__
#include <emmintrin.h>

static float compute_nb_energy_sse2(
    const float *x, const float *y, const float *z,
    const float *q, size_t n, float cutoff_sq
) {
    float energy = 0.0f;
    __m128 cutoff_sq_v = _mm_set1_ps(cutoff_sq);
    
    for (size_t i = 0; i < n; i++) {
        __m128 xi = _mm_set1_ps(x[i]);
        __m128 yi = _mm_set1_ps(y[i]);
        __m128 zi = _mm_set1_ps(z[i]);
        __m128 qi = _mm_set1_ps(q[i]);
        
        for (size_t j = i + 1; j < n; j += 4) {
            size_t remaining = (j + 4 <= n) ? 4 : (n - j);
            
            __m128 xj = _mm_loadu_ps(&x[j]);
            __m128 yj = _mm_loadu_ps(&y[j]);
            __m128 zj = _mm_loadu_ps(&z[j]);
            __m128 qj = _mm_loadu_ps(&q[j]);
            
            __m128 dx = _mm_sub_ps(xj, xi);
            __m128 dy = _mm_sub_ps(yj, yi);
            __m128 dz = _mm_sub_ps(zj, zi);
            
            __m128 r2 = _mm_add_ps(_mm_mul_ps(dx, dx),
                        _mm_add_ps(_mm_mul_ps(dy, dy),
                                   _mm_mul_ps(dz, dz)));
            
            __m128 mask = _mm_cmplt_ps(r2, cutoff_sq_v);
            
            if (_mm_movemask_ps(mask) != 0) {
                /* 计算LJ和库仑能量 */
                float r2_arr[4], qj_arr[4];
                _mm_storeu_ps(r2_arr, r2);
                _mm_storeu_ps(qj_arr, qj);
                
                for (int k = 0; k < remaining; k++) {
                    if (r2_arr[k] < cutoff_sq) {
                        float r = sqrtf(r2_arr[k]);
                        energy += compute_lj_energy_scalar(r2_arr[k], C6, C12);
                        energy += compute_coulomb_energy_scalar(q[i], qj_arr[k], r);
                    }
                }
            }
        }
    }
    
    return energy;
}
#endif

/* ========== AVX实现 ========== */

#ifdef __AVX__
#include <immintrin.h>

static float compute_nb_energy_avx(
    const float *x, const float *y, const float *z,
    const float *q, size_t n, float cutoff_sq
) {
    float energy = 0.0f;
    __m256 cutoff_sq_v = _mm256_set1_ps(cutoff_sq);
    
    for (size_t i = 0; i < n; i++) {
        __m256 xi = _mm256_set1_ps(x[i]);
        __m256 yi = _mm256_set1_ps(y[i]);
        __m256 zi = _mm256_set1_ps(z[i]);
        __m256 qi = _mm256_set1_ps(q[i]);
        
        for (size_t j = i + 1; j < n; j += 8) {
            size_t remaining = (j + 8 <= n) ? 8 : (n - j);
            
            __m256 xj = _mm256_loadu_ps(&x[j]);
            __m256 yj = _mm256_loadu_ps(&y[j]);
            __m256 zj = _mm256_loadu_ps(&z[j]);
            __m256 qj = _mm256_loadu_ps(&q[j]);
            
            __m256 dx = _mm256_sub_ps(xj, xi);
            __m256 dy = _mm256_sub_ps(yj, yi);
            __m256 dz = _mm256_sub_ps(zj, zi);
            
            __m256 r2 = _mm256_add_ps(_mm256_mul_ps(dx, dx),
                        _mm256_add_ps(_mm256_mul_ps(dy, dy),
                                      _mm256_mul_ps(dz, dz)));
            
            __m256 mask = _mm256_cmp_ps(r2, cutoff_sq_v, _CMP_LT_OS);
            
            if (_mm256_movemask_ps(mask) != 0) {
                float r2_arr[8], qj_arr[8];
                _mm256_storeu_ps(r2_arr, r2);
                _mm256_storeu_ps(qj_arr, qj);
                
                for (int k = 0; k < remaining; k++) {
                    if (r2_arr[k] < cutoff_sq) {
                        float r = sqrtf(r2_arr[k]);
                        energy += compute_lj_energy_scalar(r2_arr[k], C6, C12);
                        energy += compute_coulomb_energy_scalar(q[i], qj_arr[k], r);
                    }
                }
            }
        }
    }
    
    return energy;
}
#endif

/* ========== 基准测试 ========== */

typedef struct {
    const char *name;
    float (*func)(const float*, const float*, const float*, const float*, size_t, float);
    int supported;
    double time_ms;
    double speedup;
    float energy;
} benchmark_result_t;

/* 修复timer_t冲突 */
#undef timer_t
#define my_timer_t bench_timer_t

int main(int argc, char **argv) {
    test_init("Non-bonded SIMD Benchmark");
    
    /* 分配内存 */
    float *x = NULL, *y = NULL, *z = NULL, *q = NULL;
    ALIGNED_MALLOC(x, PARTICLE_COUNT * sizeof(float), SIMD_ALIGN);
    ALIGNED_MALLOC(y, PARTICLE_COUNT * sizeof(float), SIMD_ALIGN);
    ALIGNED_MALLOC(z, PARTICLE_COUNT * sizeof(float), SIMD_ALIGN);
    ALIGNED_MALLOC(q, PARTICLE_COUNT * sizeof(float), SIMD_ALIGN);
    
    if (!x || !y || !z || !q) {
        printf("ERROR: Failed to allocate memory\n");
        return 1;
    }
    
    /* 生成测试数据 */
    printf("Generating test data (%d particles)...\n", PARTICLE_COUNT);
    generate_random_positions(x, y, z, PARTICLE_COUNT, BOX_SIZE);
    generate_random_charges(q, PARTICLE_COUNT, -1.0f, 1.0f);
    
    /* 基准测试 */
    benchmark_result_t results[] = {
        {"Scalar", compute_nb_energy_scalar, 1, 0, 1.0, 0},
#ifdef __SSE2__
        {"SSE2", compute_nb_energy_sse2, 1, 0, 1.0, 0},
#endif
#ifdef __AVX__
        {"AVX", compute_nb_energy_avx, 1, 0, 1.0, 0},
#endif
        {NULL, NULL, 0, 0, 0, 0}
    };
    
    int num_bench = sizeof(results) / sizeof(results[0]) - 1;
    
    printf("\nRunning benchmarks (%d iterations)...\n\n", BENCH_ITER);
    
    /* 预热 */
    printf("Warmup...\n");
    for (int i = 0; i < WARMUP_ITER; i++) {
        compute_nb_energy_scalar(x, y, z, q, PARTICLE_COUNT, CUTOFF_SQ);
    }
    
    /* 运行基准测试 */
    for (int b = 0; b < num_bench; b++) {
        if (!results[b].supported) continue;
        
        printf("[BENCHMARK] %s...\n", results[b].name);
        
        my_timer_t t;
        timer_start(&t);
        
        float energy = 0.0f;
        for (int i = 0; i < BENCH_ITER; i++) {
            energy = results[b].func(x, y, z, q, PARTICLE_COUNT, CUTOFF_SQ);
        }
        
        results[b].time_ms = timer_end(&t) / BENCH_ITER;
        results[b].energy = energy;
    }
    
    /* 计算加速比 */
    for (int b = 0; b < num_bench; b++) {
        if (results[b].supported && b > 0) {
            results[b].speedup = results[0].time_ms / results[b].time_ms;
        }
    }
    
    /* 输出结果 */
    printf("\n");
    printf("========================================\n");
    printf("Benchmark Results\n");
    printf("========================================\n");
    printf("%-10s %12s %12s %12s\n", "Method", "Time (ms)", "Speedup", "Energy");
    printf("----------------------------------------\n");
    
    for (int b = 0; b < num_bench; b++) {
        if (results[b].supported) {
            printf("%-10s %12.3f %12.2fx %12.6f\n",
                   results[b].name,
                   results[b].time_ms,
                   results[b].speedup,
                   results[b].energy);
        }
    }
    
    printf("========================================\n");
    
    /* 验证结果一致性 */
    printf("\nVerifying result consistency...\n");
    for (int b = 1; b < num_bench; b++) {
        if (results[b].supported) {
            float diff = fabsf(results[b].energy - results[0].energy);
            float rel_err = diff / fabsf(results[0].energy);
            printf("  %s vs Scalar: relative error = %.6e\n",
                   results[b].name, rel_err);
        }
    }
    
    /* 清理 */
    ALIGNED_FREE(x);
    ALIGNED_FREE(y);
    ALIGNED_FREE(z);
    ALIGNED_FREE(q);
    
    test_summary();
    return 0;
}
