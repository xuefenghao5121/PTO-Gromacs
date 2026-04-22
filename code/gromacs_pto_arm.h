/*
 * GROMACS PTO (Parallel Tile Operation) for ARM SVE
 * 
 * 非键相互作用优化 - 头文件
 * 
 * 功能：
 * - Tile划分设计
 * - 全流程算子融合
 * - SVE向量化
 */

#ifndef GROMACS_PTO_ARM_H
#define GROMACS_PTO_ARM_H

#include <stdint.h>
#include <stdbool.h>
#include <arm_sve.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 版本信息
 */
#define GROMACS_PTO_ARM_VERSION_MAJOR 1
#define GROMACS_PTO_ARM_VERSION_MINOR 0
#define GROMACS_PTO_ARM_VERSION_PATCH 0

/*
 * 配置参数 - 可根据不同硬件自动调优
 */
typedef struct {
    int tile_size_atoms;        /* 每个Tile处理的原子数 默认: 64 */
    int tile_size_cache_kb;     /* Tile目标缓存大小 (KB) 默认: 512 (匹配L2) */
    bool enable_sve;            /* 启用SVE向量化 默认: true */
    bool enable_fusion;         /* 启用全流程融合 默认: true */
    bool verbose;               /* 调试输出 */
} gmx_pto_config_t;

/*
 * Tile描述符 - PTO核心数据结构
 * 
 * 每个Tile代表一个独立可并行的计算块
 */
typedef struct {
    int tile_id;                /* Tile唯一ID */
    int start_atom;             /* 起始原子索引 */
    int num_atoms;              /* 此Tile中原子数 */
    int *atom_indices;          /* 原子索引数组 [num_atoms] */
    
    /* 空间范围 - 用于邻域搜索 */
    float min_coord[3];
    float max_coord[3];
    
    /* 存储 */
    bool forces_computed;       /* 是否已计算力 */
} gmx_pto_tile_t;

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
} gmx_pto_nonbonded_params_t;

/*
 * 原子数据结构 - 坐标和力
 * 使用SVE可变长度向量
 */
typedef struct {
    int num_atoms;              /* 原子总数 */
    float *x;                   /* 坐标, 布局: [num_atoms][3] 连续存储 */
    float *f;                   /* 力, 布局: [num_atoms][3] 连续存储 */
} gmx_pto_atom_data_t;

/*
 * 邻居列表 - 按Tile分组的邻居对
 */
typedef struct {
    int tile_i;                 /* 第一个Tile */
    int tile_j;                 /* 第二个Tile */
    int num_pairs;              /* 原子对数量 */
    int *pairs;                 /* 原子对索引 (local_i, local_j) */
} gmx_pto_neighbor_pair_t;

/*
 * PTO非键计算上下文
 */
typedef struct {
    gmx_pto_config_t config;
    int num_total_atoms;
    int num_tiles;
    gmx_pto_tile_t *tiles;      /* Tile数组 [num_tiles] */
    int num_neighbor_pairs;
    gmx_pto_neighbor_pair_t *neighbor_pairs;  /* 邻居Tile对 */
    gmx_pto_nonbonded_params_t params;
    
    /* SVE状态 */
    char sve_predicate_reserved[64]; /* SVE谓词掩码 */
} gmx_pto_nonbonded_context_t;

/*
 * 函数声明
 */

/* ===== 1. Tile划分 ===== */

/**
 * 初始化PTO配置为默认值
 */
void gmx_pto_config_init(gmx_pto_config_t *config);

/**
 * 根据原子总数和缓存大小自动计算最优Tile大小
 * 返回推荐的Tile大小（原子数）
 */
int gmx_pto_auto_tile_size(int total_atoms, int cache_size_kb);

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
int gmx_pto_create_tiling(int total_atoms, const float *coords, 
                           const gmx_pto_config_t *config,
                           gmx_pto_nonbonded_context_t *context);

/**
 * 释放Tile划分资源
 */
void gmx_pto_destroy_tiling(gmx_pto_nonbonded_context_t *context);

/* ===== 2. 邻居对构建 ===== */

/**
 * 构建Tile邻居对 - 基于空间距离
 * 
 * 在PTO中，邻居搜索在Tile层级进行，减少细粒度遍历
 */
int gmx_pto_build_neighbor_pairs(gmx_pto_nonbonded_context_t *context, 
                                 const float *coords,
                                 float cutoff);

/* ===== 3. SVE向量化核心计算 ===== */

/**
 * 使用SVE向量化计算一对原子间距离平方
 * 输入: 两个原子坐标向量
 * 返回: dx^2 + dy^2 + dz^2 (SVE向量)
 */
svfloat32_t gmx_pto_sve_distance_sq(svfloat32_t x1, svfloat32_t y1, svfloat32_t z1,
                                     svfloat32_t x2, svfloat32_t y2, svfloat32_t z2);

/**
 * SVE计算LJ范德华力和能量
 */
void gmx_pto_sve_lj_force(svfloat32_t rsq, svfloat32_t eps_ij, svfloat32_t sigma_ij,
                          svfloat32_t *f_force_out, svfloat32_t *energy_out);

/**
 * SVE计算短程静电力
 */
void gmx_pto_sve_coulomb_force(svfloat32_t rsq, svfloat32_t qq, svfloat32_t kappa,
                               svfloat32_t *f_force_out, svfloat32_t *energy_out);

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
 * 所有中间结果保留在寄存器/Tile缓存，不写回内存
 * 
 * 参数:
 *   context - PTO上下文
 *   atom_data - 原子坐标和力
 * 
 * 返回: 0成功, <0错误
 */
int gmx_pto_nonbonded_compute_fused(gmx_pto_nonbonded_context_t *context,
                                     gmx_pto_atom_data_t *atom_data);

/**
 * 单个Tile的融合计算 - 供并行调用
 */
void gmx_pto_nonbonded_compute_tile(gmx_pto_nonbonded_context_t *context,
                                    gmx_pto_atom_data_t *atom_data,
                                    int tile_idx);

/**
 * 计算Tile对之间的相互作用 - SVE向量化版本
 */
void gmx_pto_sve_compute_pair(gmx_pto_nonbonded_context_t *context,
                              gmx_pto_atom_data_t *atom_data,
                              gmx_pto_tile_t *tile_i,
                              gmx_pto_tile_t *tile_j);

/* ===== 5. 工具函数 ===== */

/**
 * 获取当前SVE向量长度（bits）
 */
int gmx_pto_get_sve_vector_length_bits(void);

/**
 * 获取当前SVE向量长度（float个数）
 */
int gmx_pto_get_sve_vector_length_floats(void);

/**
 * 打印配置信息
 */
void gmx_pto_print_info(const gmx_pto_nonbonded_context_t *context);

/* ===== 6. 新增函数 - Tile 工具 ===== */

/**
 * PTOTile - 动态分配Tile结构，用于存储单个Tile的坐标和力
 */
typedef struct pto_tile {
    int n_atoms;          // 原子数
    int capacity;         // 分配容量
    float (*coords)[3];   // 坐标 [n_atoms][3] 动态分配
    float (*forces)[3];   // 力 [n_atoms][3] 动态分配
} PTOTile;

/**
 * Tile分区结构 - 用于自适应划分
 */
typedef struct pto_tile_partition {
    int n_tiles;                  // Tile数量
    int capacity;                 // 分配容量
    PTOTile *tiles;               // Tile数组
} PTOTilePartition;

/**
 * 创建Tile（动态分配）
 * @param n_atoms 原子数
 * @return 新建Tile，失败返回NULL
 */
PTOTile* pto_tile_create(int n_atoms);

/**
 * 销毁Tile
 */
void pto_tile_destroy(PTOTile* tile);

/**
 * 检查Tile是否能放入指定大小缓存
 * @param tile_size Tile原子数
 * @param cache_size_kb 缓存大小 (KB)
 * @return 1如果能放，0放不下
 */
int pto_check_tile_fits_in_cache(int tile_size, int cache_size_kb);

/**
 * 自适应Tile划分 - 密度感知负载均衡
 * @param coords 原子坐标数组 [n_atoms*3]
 * @param n_atoms 总原子数
 * @param box 模拟盒子 [3][3]
 * @param target_tile_size 目标Tile大小（原子数）
 * @param partition 输出分区结果
 * @return 0成功，<0错误
 */
int pto_adaptive_tile_partition(const float *coords, int n_atoms, 
                                 float box[3][3], int target_tile_size,
                                 PTOTilePartition *partition);

/**
 * 销毁Tile分区
 */
void pto_partition_destroy(PTOTilePartition *partition);

/**
 * 最小图像约定 - 处理周期性边界条件
 * @param dx 输入输出距离，会被修正
 * @param box_length 盒子边长
 * @param box_half 盒子半长
 */
void pto_minimum_image(float *dx, float box_length, float box_half);

/**
 * 计算SVE需要的迭代次数
 * @param n 总元素数
 * @return 需要迭代的次数
 */
int pto_sve_iterations(int n);

/**
 * 获取SVE向量宽度（字数）
 * @return 每个SVE向量包含的32位字数
 */
int pto_sve_vector_width_words(void);

/**
 * 单个原子对的非键相互作用计算，保证力对称
 * 用于验证牛顿第三定律
 */
void pto_nonbonded_pair_compute(float coords_i[3], float coords_j[3], 
                                 float force_i[3], float force_j[3],
                                 float sigma, float epsilon);

/**
 * 计算LJ能量（参考实现）
 */
float pto_lj_energy(float r, float sigma, float epsilon);

#ifdef __cplusplus
}
#endif

#endif /* GROMACS_PTO_ARM_H */
