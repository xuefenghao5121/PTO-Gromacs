/**
 * @file test_framework_x86.c
 * @brief x86平台GROMACS测试框架实现 (简化版)
 */

#include "test_framework_x86.h"

/* 全局变量 */
cpu_info_t g_cpu_info = {0};
int test_count = 0, test_passed = 0, test_failed = 0, test_skipped = 0;
static const char *current_test = NULL;
static bench_timer_t test_timer;

/* CPUID辅助函数 */
static void cpuid(int leaf, int subleaf, uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx) {
#ifdef __x86_64__
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#else
    *eax = *ebx = *ecx = *edx = 0;
#endif
}

/* 检测CPU信息 */
int detect_cpu_info(cpu_info_t *info) {
    memset(info, 0, sizeof(cpu_info_t));
    
#ifdef __x86_64__
    uint32_t eax, ebx, ecx, edx;
    
    /* 获取厂商 */
    cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    memcpy(info->vendor, &ebx, 4);
    memcpy(info->vendor + 4, &edx, 4);
    memcpy(info->vendor + 8, &ecx, 4);
    info->vendor[12] = '\0';
    
    /* 获取品牌字符串 */
    uint32_t *brand = (uint32_t*)info->brand;
    cpuid(0x80000002, 0, &brand[0], &brand[1], &brand[2], &brand[3]);
    cpuid(0x80000003, 0, &brand[4], &brand[5], &brand[6], &brand[7]);
    cpuid(0x80000004, 0, &brand[8], &brand[9], &brand[10], &brand[11]);
    info->brand[48] = '\0';
    
    /* 去除前后空格 */
    char *start = info->brand;
    while (*start == ' ') start++;
    memmove(info->brand, start, strlen(start) + 1);
    
    /* 获取特征位 */
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    info->has_sse2 = (edx >> 26) & 1;
    info->has_sse4_1 = (ecx >> 19) & 1;
    info->has_avx = (ecx >> 28) & 1;
    
    /* 检查AVX2 */
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    info->has_avx2 = (ebx >> 5) & 1;
    info->has_avx512f = (ebx >> 16) & 1;
    
    /* 获取核心数 */
    info->cores = sysconf(_SC_NPROCESSORS_ONLN);
    
    /* 频率估算 */
    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "cpu MHz", 7) == 0) {
                float mhz;
                sscanf(line, "cpu MHz\t: %f", &mhz);
                info->frequency_ghz = mhz / 1000.0;
                break;
            }
        }
        fclose(fp);
    }
    
    return 0;
#else
    return -1;
#endif
}

/* 打印CPU信息 */
void print_cpu_info(const cpu_info_t *info) {
    printf("CPU Information:\n");
    printf("  Vendor:    %s\n", info->vendor);
    printf("  Brand:     %s\n", info->brand);
    printf("  Cores:     %d\n", info->cores);
    printf("  Frequency: %.2f GHz\n", info->frequency_ghz);
    printf("  SIMD:      SSE2=%d SSE4.1=%d AVX=%d AVX2=%d AVX512=%d\n",
           info->has_sse2, info->has_sse4_1, info->has_avx,
           info->has_avx2, info->has_avx512f);
}

/* 检查SIMD支持 */
int check_simd_support(int level) {
    switch (level) {
        case 0: return 1;
        case 2: return g_cpu_info.has_sse2;
        case 4: return g_cpu_info.has_sse4_1;
        case 8: return g_cpu_info.has_avx;
        case 16: return g_cpu_info.has_avx2;
        case 32: return g_cpu_info.has_avx512f;
        default: return 0;
    }
}

/* 计时器函数 */
void timer_start(bench_timer_t *t) {
    gettimeofday(&t->start, NULL);
}

double timer_end(bench_timer_t *t) {
    gettimeofday(&t->end, NULL);
    return (t->end.tv_sec - t->start.tv_sec) * 1000.0 +
           (t->end.tv_usec - t->start.tv_usec) / 1000.0;
}

/* 内存使用 */
size_t get_peak_memory(void) {
    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp) return 0;
    char line[256];
    size_t vmhwm = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line, "VmHWM: %zu kB", &vmhwm);
            break;
        }
    }
    fclose(fp);
    return vmhwm;
}

/* 测试框架 */
void test_init(const char *suite_name) {
    test_count = test_passed = test_failed = test_skipped = 0;
    detect_cpu_info(&g_cpu_info);
    
    printf("\n========================================\n");
    printf("Test Suite: %s\n", suite_name);
    printf("========================================\n\n");
    print_cpu_info(&g_cpu_info);
    printf("\n");
}

void test_begin(const char *test_name) {
    current_test = test_name;
    test_count++;
    timer_start(&test_timer);
    printf("[TEST %d] %s: running...\n", test_count, test_name);
}

static void test_report(const char *status, const char *msg) {
    double elapsed = timer_end(&test_timer);
    printf("[TEST %d] %s: %s (%.3f ms) - %s\n",
           test_count, current_test, status, elapsed, msg);
}

void test_pass(const char *msg) {
    test_passed++;
    test_report("PASS", msg);
}

void test_fail(const char *msg) {
    test_failed++;
    test_report("FAIL", msg);
}

void test_skip(const char *msg) {
    test_skipped++;
    test_report("SKIP", msg);
}

void test_summary(void) {
    printf("\n========================================\n");
    printf("Summary: %d total, %d passed, %d failed, %d skipped\n",
           test_count, test_passed, test_failed, test_skipped);
    printf("Memory:  %zu KB peak\n", get_peak_memory());
    printf("========================================\n");
    printf("%s\n", test_failed > 0 ? "RESULT: FAIL" : "RESULT: PASS");
}

/* 数据生成 */
void generate_random_positions(float *x, float *y, float *z, size_t n, float box_size) {
    srand(42);
    for (size_t i = 0; i < n; i++) {
        x[i] = (float)rand() / RAND_MAX * box_size;
        y[i] = (float)rand() / RAND_MAX * box_size;
        z[i] = (float)rand() / RAND_MAX * box_size;
    }
}

void generate_random_charges(float *q, size_t n, float q_min, float q_max) {
    srand(42);
    for (size_t i = 0; i < n; i++) {
        q[i] = q_min + (float)rand() / RAND_MAX * (q_max - q_min);
    }
}

/* 能量计算 */
float compute_lj_energy_scalar(float r2, float c6, float c12) {
    float r2_inv = 1.0f / r2;
    float r6_inv = r2_inv * r2_inv * r2_inv;
    return c12 * r6_inv * r6_inv - c6 * r6_inv;
}

float compute_coulomb_energy_scalar(float q1, float q2, float r) {
    const float ONE_4PI_EPS0 = 1389.354578f;
    return ONE_4PI_EPS0 * q1 * q2 / r;
}

float compute_bond_energy_scalar(float dx, float dy, float dz, float r0, float k) {
    float r = sqrtf(dx*dx + dy*dy + dz*dz);
    float dr = r - r0;
    return 0.5f * k * dr * dr;
}
