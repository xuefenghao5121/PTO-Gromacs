/**
 * @file test_framework_x86.h
 * @brief x86平台GROMACS测试框架头文件 (简化版)
 */

#ifndef TEST_FRAMEWORK_X86_H
#define TEST_FRAMEWORK_X86_H

#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef __x86_64__
#include <immintrin.h>
#include <cpuid.h>
#endif

/* 测试结果状态 */
typedef enum {
    TEST_PASS = 0,
    TEST_FAIL = 1,
    TEST_SKIP = 2,
    TEST_ERROR = 3
} test_status_t;

/* CPU信息结构 */
typedef struct {
    char vendor[16];
    char brand[64];
    int cores;
    int has_sse2;
    int has_sse4_1;
    int has_avx;
    int has_avx2;
    int has_avx512f;
    double frequency_ghz;
} cpu_info_t;

/* 全局CPU信息 */
extern cpu_info_t g_cpu_info;

/* 统计变量 */
extern int test_count, test_passed, test_failed, test_skipped;

/* 函数声明 */

/* CPU检测 */
int detect_cpu_info(cpu_info_t *info);
void print_cpu_info(const cpu_info_t *info);
int check_simd_support(int level);

/* 计时器 */
typedef struct {
    struct timeval start;
    struct timeval end;
} bench_timer_t;

void timer_start(bench_timer_t *t);
double timer_end(bench_timer_t *t);

/* 内存使用 */
size_t get_peak_memory(void);

/* 测试辅助 */
void test_init(const char *suite_name);
void test_begin(const char *test_name);
void test_pass(const char *msg);
void test_fail(const char *msg);
void test_skip(const char *msg);
void test_summary(void);

/* 数据生成 */
void generate_random_positions(float *x, float *y, float *z, size_t n, float box_size);
void generate_random_charges(float *q, size_t n, float q_min, float q_max);

/* 能量计算 */
float compute_lj_energy_scalar(float r2, float c6, float c12);
float compute_coulomb_energy_scalar(float q1, float q2, float r);
float compute_bond_energy_scalar(float dx, float dy, float dz, float r0, float k);

/* SIMD辅助宏 */
#ifdef __AVX__
#define SIMD_ALIGN 32
#else
#define SIMD_ALIGN 16
#endif

#define ALIGNED_MALLOC(ptr, size, align) \
    posix_memalign((void**)&(ptr), align, size)

#define ALIGNED_FREE(ptr) free(ptr)

#endif /* TEST_FRAMEWORK_X86_H */
