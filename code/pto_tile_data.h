/*
 * PTO Tile Data Structure - Refactored for Memory Locality
 * 
 * 核心改进：Tile内部连续存储坐标/力/LJ参数，消除gather/scatter
 * 
 * 设计原则（柱子哥方案）：
 * 1. Tile构建时一次性将全局数据打包到Tile内部连续存储
 * 2. 计算时从Tile连续内存加载，不访问全局数组
 * 3. 力累加在Tile内部缓冲区完成，计算结束后一次性写回
 */

#ifndef PTO_TILE_DATA_H
#define PTO_TILE_DATA_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Tile内部连续存储的数据结构
 * 
 * 关键：所有数据都是连续的，对SIMD友好
 * 布局: SoA (Structure of Arrays) 而非 AoS
 * 
 * 内存布局:
 *   x[0], x[1], ..., x[n-1]  <- 连续，SVE可直接加载
 *   y[0], y[1], ..., y[n-1]  <- 连续
 *   z[0], z[1], ..., z[n-1]  <- 连续
 *   fx[0], fx[1], ..., fx[n-1]
 *   fy[0], fy[1], ..., fy[n-1]
 *   fz[0], fz[1], ..., fz[n-1]
 *   lj_sigma_sq[0], ..., lj_sigma_sq[n-1]  <- 预计算sigma^2
 *   lj_eps[0], ..., lj_eps[n-1]
 *   charges[0], ..., charges[n-1]
 */
typedef struct {
    int n_atoms;          /* Tile内原子数 */
    int capacity;         /* 分配容量 */
    int *global_indices;  /* 全局原子索引 [n_atoms] */
    
    /* SoA连续存储 - 每个数组n_atoms个元素，16字节对齐 */
    float *x;             /* x坐标 [n_atoms] */
    float *y;             /* y坐标 [n_atoms] */
    float *z;             /* z坐标 [n_atoms] */
    
    float *fx;            /* x力分量 [n_atoms] - 累加缓冲区 */
    float *fy;            /* y力分量 [n_atoms] */
    float *fz;            /* z力分量 [n_atoms] */
    
    /* 预打包的LJ参数 */
    float *lj_sigma_sq;   /* sigma^2 per atom [n_atoms] */
    float *lj_eps;        /* epsilon per atom [n_atoms] */
    
    /* 电荷 */
    float *charges;       /* charge per atom [n_atoms] */
    
    /* 内存管理 */
    void *mem_block;      /* 单一内存块，一次性free */
    size_t mem_size;      /* 内存块大小 */
} pto_tile_data_t;

/*
 * Tile对交互描述符
 * 
 * 描述两个Tile之间的所有有效原子对
 * 使用局部索引（相对于各自Tile），避免全局索引查找
 */
typedef struct {
    int tile_i;           /* 第一个Tile索引 */
    int tile_j;           /* 第二个Tile索引 */
    int num_pairs;        /* 有效原子对数量 */
    
    /* 原子对索引 - 使用Tile局部索引 */
    int *local_i;         /* Tile i中的局部索引 [num_pairs] */
    int *local_j;         /* Tile j中的局部索引 [num_pairs] */
    
    /* 预计算的参数对（可选，进一步消除gather） */
    /* float *pair_eps;   // eps_ij = sqrt(eps_i * eps_j) */
    /* float *pair_sig;   // sigma_ij = (sigma_i + sigma_j) / 2 */
} pto_tile_pair_t;

/*
 * PTO Tiling上下文 - 重构版
 */
typedef struct {
    int num_tiles;
    int num_pairs;
    int total_atoms;
    float cutoff;
    float box[3];
    
    pto_tile_data_t *tiles;   /* Tile数组 [num_tiles] */
    pto_tile_pair_t *pairs;   /* Tile对数组 [num_pairs] */
} pto_tiling_ctx_t;

/* ===== API ===== */

/**
 * 创建Tile数据结构并分配内存
 */
pto_tile_data_t* pto_tile_data_create(int capacity);

/**
 * 销毁Tile数据结构
 */
void pto_tile_data_destroy(pto_tile_data_t *tile);

/**
 * 打包：从全局坐标数组复制到Tile内部连续存储
 * 
 * 这是消除gather的关键步骤：
 * - 将全局coords[global_idx*3+0] -> tile->x[local_idx]
 * - 将全局coords[global_idx*3+1] -> tile->y[local_idx]
 * - 将全局coords[global_idx*3+2] -> tile->z[local_idx]
 * - 预计算LJ参数打包
 */
void pto_tile_data_pack_coords(pto_tile_data_t *tile,
                                const float *global_coords,
                                const float *global_lj_sigma,
                                const float *global_lj_eps,
                                const float *global_charges);

/**
 * 解包：将Tile内部的力写回全局力数组
 * 
 * 这是消除scatter的关键步骤：
 * - 将tile->fx[local_idx] += global_f[global_idx*3+0]
 * 一次性完成，不是逐对scatter
 */
void pto_tile_data_unpack_forces(const pto_tile_data_t *tile,
                                  float *global_forces);

/**
 * 创建完整Tiling上下文
 * 
 * 步骤：
 * 1. 将原子按空间分块到Tile
 * 2. 打包每个Tile的坐标和参数到连续存储
 * 3. 构建Tile间邻居对
 */
int pto_tiling_create(const float *coords, int natoms,
                       const float *lj_sigma, const float *lj_eps,
                       const float *charges,
                       const float box[3], float cutoff,
                       int target_tile_size,
                       pto_tiling_ctx_t *ctx);

/**
 * 销毁Tiling上下文
 */
void pto_tiling_destroy(pto_tiling_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* PTO_TILE_DATA_H */
