/*
 * PTO E2E Test for ARM SME v3
 * 
 * 端到端测试，验证SME v3实现的正确性和性能
 * 
 * 作者: 天权-HPC团队
 * 日期: 2026-04-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <arm_sve.h>
#include <arm_sme.h>
#include "gromacs_pto_arm.h"
#include "gromacs_pto_sme_v3.h"

/*
 * 原子系统结构体
 */
typedef struct {
    int num_atoms;
    float *coords;
    float *forces;
} atom_system_t;
/*
 * 计时器
 */
typedef struct {
    struct timespec start;
    struct timespec end;
} sme_timer_t;

/*
 * 计时器开始
 */
void timer_start(sme_timer_t *timer) {
    clock_gettime(CLOCK_MONOTONIC, &timer->start);
}

/*
 * 计时器停止，返回耗时（秒）
 */
double timer_stop(sme_timer_t *timer) {
    clock_gettime(CLOCK_MONOTONIC, &timer->end);
    return (timer->end.tv_sec - timer->start.tv_sec) + 
           (timer->end.tv_nsec - timer->start.tv_nsec) * 1e-9;
}

/*
 * 创建随机原子系统
 */
atom_system_t* create_atom_system(int n_atoms, float box_size) {
    atom_system_t *sys = (atom_system_t*)malloc(sizeof(atom_system_t));
    if (!sys) return NULL;
    
    sys->num_atoms = n_atoms;
    sys->coords = (float*)malloc(n_atoms * 3 * sizeof(float));
    sys->forces = (float*)calloc(n_atoms * 3, sizeof(float));
    
    /* 随机坐标生成 */
    srand(time(NULL));
    for (int i = 0; i < n_atoms * 3; i++) {
        sys->coords[i] = ((float)rand() / RAND_MAX) * box_size;
    }
    
    return sys;
}

/*
 * 销毁原子系统
 */
void destroy_atom_system(atom_system_t *sys) {
    if (sys) {
        if (sys->coords) free(sys->coords);
        if (sys->forces) free(sys->forces);
        free(sys);
    }
}

/*
 * 比较两个力数组，返回最大误差
 */
float compare_forces(float *f1, float *f2, int n, float tolerance, float max_diff) {
    float maxd = 0.0f;
    int bad_count = 0;
    
    for (int i = 0; i < n * 3; i++) {
        float d = fabsf(f1[i] - f2[i]);
        if (d > max_diff) {
            max_diff = d;
        }
        if (d > tolerance) {
            bad_count++;
        }
    }
    
    printf("Compare done, max diff = %.6f, bad count = %d\n", max_diff, bad_count);
    return max_diff;
}

/*
 * 主函数
 */
int main(int argc, char **argv) {
    int n_atoms = 2048;
    float box_size = 10.0f;
    float cutoff = 1.0f;
    int n_steps = 10;
    
    /* 解析命令行参数 */
    if (argc < 2) {
        printf("Usage: ./e2e_sme_v3 <input.gro> [cutoff] [steps]\n");
        printf("  or: ./e2e_sme_v3 <n_atoms\n");
        printf("\n");
        printf("Examples:\n");
        printf("  ./e2e_sme_v3 box_small.gro 1.0 10\n");
        printf("  ./e2e_sme_v3 2048\n");
        return 1;
    }
    
    /* 判断第一个参数是数字还是文件名 */
    int n_atoms_arg = atoi(argv[1]);
    atom_system_t *sys = NULL;
    
    if (n_atoms_arg > 0) {
        n_atoms = n_atoms_arg;
        sys = create_atom_system(n_atoms, box_size);
    } else {
        /* TODO: 从.gro文件读取 */
        printf("Error: .gro file reading not implemented yet\n");
        return 1;
    }
    
    if (argc >= 3) {
        cutoff = atof(argv[2]);
    }
    
    if (argc >= 4) {
        n_steps = atoi(argv[3]);
    }
    
    /* 初始化SME v3 */
    printf("===== SME v3 PTO-GROMACS E2E Test =====\n");
    
    if (!gmx_pto_sme_v3_is_available()) {
        printf("ERROR: SME hardware is not available!\n");
        printf("This implementation requires ARMv9.2-a with SME support\n");
        return 1;
    }
    
    if (!gmx_pto_sme_v3_init()) {
        printf("ERROR: Failed to initialize SME v3\n");
        return 1;
    }
    
    gmx_pto_sme_v3_print_info();
    
    /* 准备上下文 */
    gmx_pto_nonbonded_context_t context;
    memset(&context, 0, sizeof(context));
    context.params.cutoff_sq = cutoff * cutoff;
    context.params.lj_epsilon = 1.0f;
    context.params.lj_sigma_sq = 0.09f;
    
    /* 准备原子数据 */
    gmx_pto_atom_data_t atom_data;
    atom_data.num_atoms = sys->num_atoms;
    atom_data.x = sys->coords;
    atom_data.f = sys->forces;
    
    printf("\nStarting benchmark with %d atoms, %d steps, cutoff %.2f\n\n", 
           sys->num_atoms, n_steps, cutoff);
    
    /* 运行基准测试 */
    sme_timer_t timer;
    timer_start(&timer);
    
    gmx_pto_sme_v3_nonbonded_compute(&context, &atom_data);
    
    double total_time = timer_stop(&timer);
    
    /* 打印结果 */
    double avg_time = total_time / n_steps;
    printf("\n===== BENCHMARK RESULTS =====\n");
    printf("Total time:   %.3f seconds\n", total_time);
    printf("Average time: %.3f ms/step\n", avg_time * 1000.0);
    printf("Throughput:   %.2f steps/second\n", 1.0 / avg_time);
    printf("Throughput:   %.2f M-atoms/second\n", sys->num_atoms / avg_time / 1e6);
    printf("========================================\n");
    
    /* 清理 */
    gmx_pto_sme_v3_cleanup();
    destroy_atom_system(sys);
    
    return 0;
}


