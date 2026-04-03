/**
 * @file test_unit_x86.c
 * @brief x86平台单元测试 (简化版)
 */

#define _DEFAULT_SOURCE
#include "test_framework_x86.h"

/* ========== CPU检测测试 ========== */

void test_cpu_detection(void) {
    test_begin("CPU Detection");
    
    cpu_info_t info;
    int ret = detect_cpu_info(&info);
    
    if (ret != 0) {
        test_fail("detect_cpu_info failed");
        return;
    }
    if (info.cores <= 0) {
        test_fail("Invalid core count");
        return;
    }
    if (info.vendor[0] == '\0') {
        test_fail("Vendor string empty");
        return;
    }
    if (info.brand[0] == '\0') {
        test_fail("Brand string empty");
        return;
    }
    if (!info.has_sse2) {
        test_fail("SSE2 not supported on x86_64");
        return;
    }
    
    test_pass("All CPU fields populated correctly");
}

void test_simd_detection(void) {
    test_begin("SIMD Detection");
    
    if (!check_simd_support(0)) {
        test_fail("Scalar should always be supported");
        return;
    }
    if (check_simd_support(2) != g_cpu_info.has_sse2) {
        test_fail("SSE2 mismatch");
        return;
    }
    if (check_simd_support(8) != g_cpu_info.has_avx) {
        test_fail("AVX mismatch");
        return;
    }
    
    char msg[128];
    snprintf(msg, sizeof(msg), "SSE2=%d, AVX=%d, AVX2=%d, AVX512=%d",
             g_cpu_info.has_sse2, g_cpu_info.has_avx,
             g_cpu_info.has_avx2, g_cpu_info.has_avx512f);
    test_pass(msg);
}

/* ========== 计时器测试 ========== */

void test_timer_accuracy(void) {
    test_begin("Timer Accuracy");
    
    bench_timer_t t;
    timer_start(&t);
    usleep(10000);  /* 10ms */
    double elapsed = timer_end(&t);
    
    if (elapsed < 9.0 || elapsed > 50.0) {
        char msg[64];
        snprintf(msg, sizeof(msg), "Timer accuracy issue: %.3f ms", elapsed);
        test_fail(msg);
        return;
    }
    
    char msg[64];
    snprintf(msg, sizeof(msg), "Slept 10ms, measured %.3f ms", elapsed);
    test_pass(msg);
}

/* ========== 能量计算测试 ========== */

void test_lj_energy(void) {
    test_begin("LJ Energy Calculation");
    
    float sigma = 1.0f, epsilon = 1.0f;
    float c6 = 4.0f * epsilon * powf(sigma, 6);
    float c12 = 4.0f * epsilon * powf(sigma, 12);
    float r_min = powf(2.0f, 1.0f/6.0f) * sigma;
    float r2_min = r_min * r_min;
    
    float energy = compute_lj_energy_scalar(r2_min, c6, c12);
    
    if (fabsf(energy - (-epsilon)) > 1e-5) {
        char msg[64];
        snprintf(msg, sizeof(msg), "LJ minimum: got %.6f, expected -1.0", energy);
        test_fail(msg);
        return;
    }
    
    test_pass("LJ minimum verified");
}

void test_coulomb_energy(void) {
    test_begin("Coulomb Energy Calculation");
    
    float q1 = 1.0f, q2 = 1.0f, r = 1.0f;
    float energy = compute_coulomb_energy_scalar(q1, q2, r);
    float expected = 1389.354578f;
    
    if (fabsf(energy - expected) > 0.01) {
        char msg[64];
        snprintf(msg, sizeof(msg), "Coulomb: got %.6f, expected %.6f", energy, expected);
        test_fail(msg);
        return;
    }
    
    test_pass("Coulomb energy verified");
}

void test_bond_energy(void) {
    test_begin("Bond Energy Calculation");
    
    float r0 = 0.15f, k = 1000.0f;
    
    /* At equilibrium */
    float energy = compute_bond_energy_scalar(r0, 0, 0, r0, k);
    if (fabsf(energy) > 1e-6) {
        test_fail("Bond not at equilibrium");
        return;
    }
    
    /* Stretched */
    energy = compute_bond_energy_scalar(2*r0, 0, 0, r0, k);
    float expected = 0.5f * k * r0 * r0;
    if (fabsf(energy - expected) > 0.1) {
        test_fail("Bond stretched incorrect");
        return;
    }
    
    test_pass("Bond energy verified");
}

/* ========== 数据生成测试 ========== */

void test_position_generation(void) {
    test_begin("Position Generation");
    
    const size_t n = 1000;
    float x[n], y[n], z[n];
    float box = 5.0f;
    
    generate_random_positions(x, y, z, n, box);
    
    int out_of_range = 0;
    for (size_t i = 0; i < n; i++) {
        if (x[i] < 0 || x[i] > box ||
            y[i] < 0 || y[i] > box ||
            z[i] < 0 || z[i] > box) {
            out_of_range++;
        }
    }
    
    if (out_of_range > 0) {
        test_fail("Positions out of range");
        return;
    }
    
    char msg[64];
    snprintf(msg, sizeof(msg), "Generated %zu positions in [0, %.1f]", n, box);
    test_pass(msg);
}

void test_charge_generation(void) {
    test_begin("Charge Generation");
    
    const size_t n = 1000;
    float q[n];
    float q_min = -1.0f, q_max = 1.0f;
    
    generate_random_charges(q, n, q_min, q_max);
    
    int out_of_range = 0;
    for (size_t i = 0; i < n; i++) {
        if (q[i] < q_min || q[i] > q_max) {
            out_of_range++;
        }
    }
    
    if (out_of_range > 0) {
        test_fail("Charges out of range");
        return;
    }
    
    test_pass("All charges in valid range");
}

/* ========== 内存测试 ========== */

void test_memory_tracking(void) {
    test_begin("Memory Tracking");
    
    size_t before = get_peak_memory();
    
    size_t alloc_size = 10 * 1024 * 1024;
    char *buf = malloc(alloc_size);
    if (!buf) {
        test_skip("Allocation failed");
        return;
    }
    memset(buf, 0, alloc_size);
    
    size_t after = get_peak_memory();
    free(buf);
    
    char msg[64];
    snprintf(msg, sizeof(msg), "Memory delta: %zu KB", after > before ? after - before : 0);
    test_pass(msg);
}

/* ========== 主函数 ========== */

int main(void) {
    test_init("x86 Unit Tests");
    
    test_cpu_detection();
    test_simd_detection();
    test_timer_accuracy();
    test_lj_energy();
    test_coulomb_energy();
    test_bond_energy();
    test_position_generation();
    test_charge_generation();
    test_memory_tracking();
    
    test_summary();
    return test_failed > 0 ? 1 : 0;
}
