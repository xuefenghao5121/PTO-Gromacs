/*
 * GROMACS PTO (Parallel Tile Operation) for x86 AVX/AVX2
 * 
 * 方案一：非键相互作用优化 - 头文件（x86 版本）
 * 
 * 功能：
 * - Tile划分设计（适配 x86 L1/L2/L3 缓存层次）
 * - 全流程算子融合（消除中间内存写回）
 * - AVX/AVX2 向量化支持
 * 
 * 移植自 ARM 版本，核心原理一致：
 * - ARM SVE → x86 AVX/AVX2
 * - ARM SME Tile 寄存器 → x86 向量寄存器
 */

#ifndef GROMACS_PTO_X86_H
#define GROMACS_PTO_X86_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 版本信息
 */
#define GROMACS_PTO_X86_VERSION_MAJOR 1
#define GROMACS_PTO_X86_VERSION_MINOR 0
#define GROMACS_PTO_X86_VERSION_PATCH 0

/*
 * x86 缓存层次常量（典型值）
 */
#define GMX_PTO_X86_L1_CACHE_KB      32    /* L1 数据缓存 */
#define GMX_PTO_X86_L2_CACHE_KB      256   /* L2 缓存 */
#define GMX_PTO_X86_L3_CACHE_KB      8192  /* L3 缓存 */
#define GMX_PTO_X86_DEFAULT_TILE_KB  256   /* 默认 Tile 大小（适配 L2） */

/*
 * SIMD 宽度
 */
#ifdef __AVX__
#define GMX_PTO_X86_VECTOR_WIDTH 8    /* AVX: 256-bit = 8 floats */
#else
#define GMX_PTO_X86_VECTOR_WIDTH 4    /* SSE: 128-bit = 4 floats */
#endif

/*
 * 配置参数 - 可根据不同硬件自动调优
 */
typedef struct {
    int tile_size_atoms;        /* 每个Tile处理的原子数 默认: 64 */
    int tile_size_cache_kb;     /* Tile目标缓存大小 (KB) 默认: 256 (匹配L2) */
    bool enable_avx;            /* 启用AVX向量化 默认: true */
    bool enable_avx2;           /* 启用AVX2 FMA 默认: true */
    bool enable_fusion;         /* 启用全流程融合 默认: true */
    bool verbose;               /* 调试输出 */
} gmx_pto_config_x86_t;

/*
 * Tile描述符 - PTO核心数据结构
 * 
 * 每个Tile代表一个独立可并行的计算块
 * 在x86中，Tile数据保存在L2缓存中
 */
typedef struct {
    int tile_id;                /* Tile唯一ID */
    int start_atom;             /* 起始原子索引 */
    int num_atoms;              /* 此Tile中原子数 */
    int *atom_indices;          /* 原子索引数组 [num_atoms] */
    
    /* 空间范围 - 用于邻域搜索 */
    float min_coord[3];
    float max_coord[3];
    
    /* 结果缓存 */
    bool forces_computed;       /* 是否已计算力 */
} gmx_pto_tile_x86_t;

/*
 * 非键相互作用参数
 */
typedef struct {
    float cutoff_sq;            /* 截断距离平方 */
    float epsilon_r;            /* 介电常数 */
    float rf_kappa;             /* 反应场因子 */
    
    /* LJ参数 */
    float *lj_epsilon;          /* epsilon per atom type */
    float *lj_sigma;            /* sigma per atom type */
    
    /* 电荷 */
    float *charges;             /* 每个原子的电荷 */
} gmx_pto_nonbonded_params_x86_t;

/*
 * 原子数据结构 - 坐标和力
 * 使用AoS布局（与GROMACS兼容）
 */
typedef struct {
    int num_atoms;              /* 原子总数 */
    float *x;                   /* 坐标, 布局: [num_atoms][3] 连续存储 */
    float *f;                   /* 力, 布局: [num_atoms][3] 连续存储 */
} gmx_pto_atom_data_x86_t;

/*
 * 邻居列表 - 按Tile分组的邻居对
 */
typedef struct {
    int tile_i;                 /* 第一个Tile */
    int tile_j;                 /* 第二个Tile */
    int num_pairs;              /* 原子对数量 */
    int *pairs;                 /* 原子对索引 (local_i, local_j) */
} gmx_pto_neighbor_pair_x86_t;

/*
 * PTO非键计算上下文（x86 版本）
 */
typedef struct {
    gmx_pto_config_x86_t config;
    int num_total_atoms;
    int num_tiles;
    gmx_pto_tile_x86_t *tiles;  /* Tile数组 [num_tiles] */
    int num_neighbor_pairs;
    gmx_pto_neighbor_pair_x86_t *neighbor_pairs;  /* 邻居Tile对 */
    gmx_pto_nonbonded_params_x86_t params;
    
    /* x86 SIMD 状态 */
    bool avx_enabled;
    bool avx2_enabled;
} gmx_pto_nonbonded_context_x86_t;

/*
 * 函数声明
 */

/* ===== 1. Tile划分 ===== */

/**
 * 初始化PTO配置为默认值
 */
void gmx_pto_config_x86_init(gmx_pto_config_x86_t *config);

/**
 * 根据原子总数和缓存大小自动计算最优Tile大小
 * 返回推荐的Tile大小（原子数）
 */
int gmx_pto_auto_tile_size_x86(int total_atoms, int cache_size_kb);

/**
 * 创建Tile划分 - 空间填充曲线分块
 * 
 * 参数:
 *   total_atoms - 总原子数
 *   coords - 原子坐标 [num_atoms*3]
 *   config - PTO配置
 *   context - 输出上下文
 * 
 * 返回: 0成功, <0错误
 */
int gmx_pto_create_tiling_x86(int total_atoms, const float *coords, 
                               const gmx_pto_config_x86_t *config,
                               gmx_pto_nonbonded_context_x86_t *context);

/**
 * 释放Tile划分资源
 */
void gmx_pto_destroy_tiling_x86(gmx_pto_nonbonded_context_x86_t *context);

/* ===== 2. 邻居对构建 ===== */

/**
 * 构建Tile邻居对 - 基于空间距离
 */
int gmx_pto_build_neighbor_pairs_x86(gmx_pto_nonbonded_context_x86_t *context, 
                                     const float *coords,
                                     float cutoff);

/* ===== 3. AVX/AVX2 向量化核心计算 ===== */

/**
 * AVX计算距离平方
 * 输入: 两个原子坐标数组
 * 输出: 距离平方数组
 */
void gmx_pto_avx_distance_sq(const float *x1, const float *y1, const float *z1,
                              const float *x2, const float *y2, const float *z2,
                              float *rsq_out, int count);

/**
 * AVX计算LJ范德华力和能量
 */
void gmx_pto_avx_lj_force(const float *rsq, const float *eps_ij, const float *sigma_ij,
                          int count, float *f_force_out, float *energy_out);

/**
 * AVX计算短程静电力
 */
void gmx_pto_avx_coulomb_force(const float *rsq, const float *qq, const float *kappa,
                               int count, float *f_force_out, float *energy_out);

/**
 * AVX 融合计算 Tile 对之间的相互作用
 */
void gmx_pto_avx_compute_pair(gmx_pto_nonbonded_context_x86_t *context,
                              gmx_pto_atom_data_x86_t *atom_data,
                              gmx_pto_tile_x86_t *tile_i,
                              gmx_pto_tile_x86_t *tile_j);

/* ===== 4. 融合全流程计算 - 主入口 ===== */

/**
 * PTO融合非键相互作用计算 - 整个流程在单个函数内完成
 * 
 * 融合了:
 * - 坐标加载
 * - 邻域对遍历
 * - LJ力计算
 * - 静电短程力计算
 * - 力累加写回
 * 
 * 所有中间结果保留在向量寄存器，不写回内存
 * 
 * 参数:
 *   context - PTO上下文
 *   atom_data - 原子坐标和力
 * 
 * 返回: 0成功, <0错误
 */
int gmx_pto_nonbonded_compute_fused_x86(gmx_pto_nonbonded_context_x86_t *context,
                                         gmx_pto_atom_data_x86_t *atom_data);

/**
 * 单个Tile的融合计算 - 供并行调用
 */
void gmx_pto_nonbonded_compute_tile_x86(gmx_pto_nonbonded_context_x86_t *context,
                                        gmx_pto_atom_data_x86_t *atom_data,
                                        int tile_idx);

/* ===== 5. 工具函数 ===== */

/**
 * 检查 CPU 是否支持 AVX
 */
bool gmx_pto_check_avx_support(void);

/**
 * 检查 CPU 是否支持 AVX2
 */
bool gmx_pto_check_avx2_support(void);

/**
 * 获取AVX向量宽度（float个数）
 */
int gmx_pto_get_avx_vector_width(void);

/**
 * 打印配置信息
 */
void gmx_pto_print_info_x86(const gmx_pto_nonbonded_context_x86_t *context);

/* ===== 6. 兼容 ARM 版本的工具函数 ===== */

/**
 * 检查Tile是否能放入指定大小缓存
 */
int pto_check_tile_fits_in_cache_x86(int tile_size, int cache_size_kb);

/**
 * 最小图像约定 - 处理周期性边界条件
 */
void pto_minimum_image_x86(float *dx, float box_length, float box_half);

/**
 * 单个原子对的非键相互作用计算，对称力累加保证牛顿第三定律精确成立
 */
void pto_nonbonded_pair_compute_x86(float coords_i[3], float coords_j[3], 
                                     float force_i[3], float force_j[3],
                                     float sigma, float epsilon);

/**
 * 计算LJ能量（参考实现）
 */
float pto_lj_energy_x86(float r, float sigma, float epsilon);

/* ===== 7. PyPTO 集成 ===== */

/**
 * 检查 PyPTO 是否可用
 */
bool gmx_pto_pypto_is_available(void);

/**
 * 初始化 PyPTO
 * 返回: 0成功, <0错误
 */
int gmx_pto_pypto_init(void);

/**
 * 清理 PyPTO 资源
 */
void gmx_pto_pypto_cleanup(void);

/**
 * 获取 PyPTO 版本信息
 * 返回: 版本字符串（需要调用者 free()）
 */
char* gmx_pto_pypto_get_version(void);

/**
 * 使用 PyPTO 创建优化的 Tile 划分
 */
int gmx_pto_pypto_create_tiling(int total_atoms, const float *coords,
                                 gmx_pto_config_x86_t *config,
                                 gmx_pto_nonbonded_context_x86_t *context);

/**
 * PyPTO 融合非键相互作用计算
 * 
 * 参数:
 *   context - PTO 上下文
 *   atom_data - 原子数据
 *   use_pypto - 是否使用 PyPTO（true=PyPTO, false=原生C）
 * 
 * 返回: 0成功, <0错误
 */
int gmx_pto_pypto_fused_compute(gmx_pto_nonbonded_context_x86_t *context,
                                 gmx_pto_atom_data_x86_t *atom_data,
                                 bool use_pypto);

#ifdef __cplusplus
}
#endif

#endif /* GROMACS_PTO_X86_H */
