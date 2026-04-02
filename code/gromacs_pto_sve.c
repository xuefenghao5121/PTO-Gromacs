/*
 * GROMACS PTO for ARM SVE/SME
 * 
 * SVE向量化核心计算实现
 * 
 * 功能:
 * - SVE距离计算
 * - SVE LJ范德华力计算
 * - SVE静电力计算
 * - 全流程融合计算
 */

#include "gromacs_pto_arm.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*
 * 计算一对原子间距离平方 - SVE向量化
 * 
 * 输入是多个原子对打包到SVE向量中
 */
svfloat32_t gmx_pto_sve_distance_sq(svfloat32_t x1, svfloat32_t y1, svfloat32_t z1,
                                     svfloat32_t x2, svfloat32_t y2, svfloat32_t z2) {
    svfloat32_t dx = svsub_f32_x(svptrue_b32(), x1, x2);
    svfloat32_t dy = svsub_f32_x(svptrue_b32(), y1, y2);
    svfloat32_t dz = svsub_f32_x(svptrue_b32(), z1, z2);
    
    svfloat32_t dx2 = svmul_f32_x(svptrue_b32(), dx, dx);
    svfloat32_t dy2 = svmul_f32_x(svptrue_b32(), dy, dy);
    svfloat32_t dz2 = svmul_f32_x(svptrue_b32(), dz, dz);
    
    return svadd_f32_x(svptrue_b32(), svadd_f32_x(svptrue_b32(), dx2, dy2), dz2);
}

/*
 * 计算LJ范德华力
 * 
 * LJ公式:
 * V(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]
 * f = -dV/dr = 24*epsilon*[2*(sigma^12/r^13) - (sigma^6/r^7)] = (24*epsilon/r^2)[2*(sigma/r)^12 - (sigma/r)^6]
 */
void gmx_pto_sve_lj_force(svfloat32_t rsq, svfloat32_t eps_ij, svfloat32_t sigma_ij,
                          svfloat32_t *f_force_out, svfloat32_t *energy_out) {
    svbool_t p = svptrue_b32();
    
    /* sigma^2 / r^2 */
    svfloat32_t sigma_sq = svmul_f32_x(p, sigma_ij, sigma_ij);
    svfloat32_t inv_rsq = svdiv_f32_x(p, svdup_f32(1.0f), rsq);
    svfloat32_t sig_inv_rsq = svmul_f32_x(p, sigma_sq, inv_rsq);
    
    /* (sigma/r)^6 = (sigma^2/r^2)^3 */
    svfloat32_t t2 = svmul_f32_x(p, sig_inv_rsq, sig_inv_rsq);  /* (sigma^2/r^2)^2 */
    svfloat32_t t6 = svmul_f32_x(p, t2, sig_inv_rsq);           /* (sigma^2/r^2)^3 = sigma^6 / r^6 */
    
    /* (sigma/r)^12 = t6^2 */
    svfloat32_t t12 = svmul_f32_x(p, t6, t6);
    
    /* 能量: V = 4*eps*(t12 - t6) */
    svfloat32_t energy = svmul_f32_x(p, svmul_f32_x(p, svdup_f32(4.0f), eps_ij),
                                     svsub_f32_x(p, t12, t6));
    *energy_out = energy;
    
    /* 力: f/r = 24*eps*(2*t12 - t6) / r^2 */
    svfloat32_t term = svsub_f32_x(p, svmul_f32_x(p, svdup_f32(2.0f), t12), t6);
    svfloat32_t f_over_r = svmul_f32_x(p, svmul_f32_x(p, svdup_f32(24.0f), eps_ij),
                                       svmul_f32_x(p, term, inv_rsq));
    *f_force_out = f_over_r;
}

/*
 * 计算短程库仑力 (反应场形式)
 * 
 * V(r) = qq / r * (1 + kappa*r) / (1 + kappa*cutoff)
 * f = -dV/dr = qq / r^2 * (1 - (kappa*r)^2/(1 + kappa*cutoff))
 */
void gmx_pto_sve_coulomb_force(svfloat32_t rsq, svfloat32_t qq, svfloat32_t kappa,
                               svfloat32_t *f_force_out, svfloat32_t *energy_out) {
    svbool_t p = svptrue_b32();
    
    svfloat32_t r = svsqrt_f32_x(p, rsq);
    svfloat32_t inv_r = svdiv_f32_x(p, svdup_f32(1.0f), r);
    svfloat32_t inv_rsq = svmul_f32_x(p, inv_r, inv_r);
    
    /* 反应场修正 */
    svfloat32_t kr = svmul_f32_x(p, kappa, r);
    svfloat32_t kr2 = svmul_f32_x(p, kr, kr);
    svfloat32_t one_kR = svadd_f32_x(p, svdup_f32(1.0f), kr);
    
    /* 能量 */
    svfloat32_t energy = svmul_f32_x(p, svmul_f32_x(p, qq, inv_r),
                                     svdiv_f32_x(p, svadd_f32_x(p, svdup_f32(1.0f), kr), one_kR));
    *energy_out = energy;
    
    /* 力: f/r */
    svfloat32_t f_over_r = svmul_f32_x(p, qq, svmul_f32_x(p, inv_rsq,
                       svdiv_f32_x(p, svsub_f32_x(p, svdup_f32(1.0f), kr2), one_kR)));
    *f_over_r = f_over_r;
    *f_force_out = f_over_r;
}

/*
 * 融合计算Tile对之间的相互作用
 * 
 * 这是核心融合函数:
 * - 加载坐标到SVE向量
 * - 计算距离
 * - 计算LJ力
 * - 计算静电力
 * - 累加力
 * 
 * 所有计算都在向量寄存器中完成，中间结果不写回内存
 */
void gmx_pto_sve_compute_pair(gmx_pto_nonbonded_context_t *context,
                              gmx_pto_atom_data_t *atom_data,
                              gmx_pto_tile_t *tile_i,
                              gmx_pto_tile_t *tile_j) {
    svbool_t all_p = svptrue_b32();
    int vl = gmx_pto_get_sve_vector_length_floats();
    float cutoff_sq = context->params.cutoff_sq;
    
    /* 获取Tile原子数据 */
    int ni = tile_i->num_atoms;
    int nj = tile_j->num_atoms;
    int *idx_i = tile_i->atom_indices;
    int *idx_j = tile_j->atom_indices;
    
    float *x = atom_data->x;
    float *f = atom_data->f;
    
    /* 遍历i原子，按SVE向量长度分组j原子 */
    for (int li = 0; li < ni; li++) {
        int gi = idx_i[li];
        svfloat32_t x1 = svdup_f32(x[gi * 3 + 0]);
        svfloat32_t y1 = svdup_f32(x[gi * 3 + 0 + 1]);
        svfloat32_t z1 = svdup_f32(x[gi * 3 + 0 + 2]);
        
        /* 累加fi的部分和，保存在向量寄存器中 */
        svfloat32_t fx_acc = svdup_f32(0.0f);
        svfloat32_t fy_acc = svdup_f32(0.0f);
        svfloat32_t fz_acc = svdup_f32(0.0f);
        
        /* 遍历j原子，按SVE向量宽度分块处理 */
        for (int lj = 0; lj < nj; lj += vl) {
            int remain = nj - lj;
            svbool_t p = svwhilelt_b32(lj, nj);
            
            /* 加载j原子坐标到SVE向量 */
            svfloat32_t x2 = svld1_f32(p, &x[idx_j[lj + 0] * 3 + 0]);
            svfloat32_t y2 = svld1_f32(p, &x[idx_j[lj + 0] * 3 + 1]);
            svfloat32_t z2 = svld1_f32(p, &x[idx_j[lj + 0] * 3 + 2]);
            
            /* 计算距离平方 */
            svfloat32_t rsq = gmx_pto_sve_distance_sq(x1, y1, z1, x2, y2, z2);
            
            /* 只处理距离小于cutoff的对 */
            svbool_t in_cutoff = svcmplt_f32(p, rsq, svdup_f32(cutoff_sq));
            
            if (svptest_any(all_p, in_cutoff)) {
                /* 这里可以优化: 预先提取类型参数，打包到SVE向量 */
                /* 简化版本: 逐个处理，核心是展示SVE向量化 */
                
                /* 计算力分量 */
                svfloat32_t dx = svsub_f32_m(in_cutoff, svdup_f32(0.0f), x1, x2);
                svfloat32_t dy = svsub_f32_m(in_cutoff, svdup_f32(0.0f), y1, y2);
                svfloat32_t dz = svsub_f32_m(in_cutoff, svdup_f32(0.0f), z1, z2);
                
                /* 获取LJ参数 - 简化版本，实际需要按类型查找 */
                /* 在完整GROMACS集成中，这会从预计算的类型对表获取 */
                svfloat32_t eps_ij = svdup_f32(0.5f);
                svfloat32_t sigma_ij = svdup_f32(0.3f);
                
                /* 获取电荷乘积 */
                svfloat32_t qq;
                if (context->params.charges != NULL) {
                    float qi = context->params.charges[gi];
                    float qj = context->params.charges[idx_j[lj]];
                    qq = svdup_f32(qi * qj);
                } else {
                    qq = svdup_f32(0.0f);
                }
                
                /* 计算LJ力 */
                svfloat32_t f_lj_over_r, energy_lj;
                gmx_pto_sve_lj_force(rsq, eps_ij, sigma_ij, &f_lj_over_r, &energy_lj);
                
                /* 计算静电力 */
                svfloat32_t f_coul_over_r = svdup_f32(0.0f);
                svfloat32_t energy_coul = svdup_f32(0.0f);
                if (svnot_z(svptest(p, qq))) {
                    gmx_pto_sve_coulomb_force(rsq, qq, context->params.rf_kappa,
                                             &f_coul_over_r, &energy_coul);
                }
                
                /* 总力除以r */
                svfloat32_t f_over_r = svadd_f32_m(in_cutoff, svdup_f32(0.0f),
                                                   f_lj_over_r, f_coul_over_r);
                
                /* 力分量 = f_over_r * 坐标差 */
                svfloat32_t fx_i = svmul_f32_m(in_cutoff, svdup_f32(0.0f), f_over_r, dx);
                svfloat32_t fy_i = svmul_f32_m(in_cutoff, svdup_f32(0.0f), f_over_r, dy);
                svfloat32_t fz_i = svmul_f32_m(in_cutoff, svdup_f32(0.0f), f_over_r, dz);
                
                /* 累加到i原子 */
                fx_acc = svadd_f32_x(all_p, fx_acc, fx_i);
                fy_acc = svadd_f32_x(all_p, fy_acc, fy_i);
                fz_acc = svadd_f32_x(all_p, fz_acc, fz_i);
                
                /* 累加到j原子 - 需要存储到j的数组 */
                /* 因为这是对j的累加，直接回存 */
                float *fj_base = &f[idx_j[lj + 0] * 3];
                svfloat32_t fx_j = svneg_f32_x(p, fx_i);
                svfloat32_t fy_j = svneg_f32_x(p, fy_i);
                svfloat32_t fz_j = svneg_f32_x(p, fz_i);
                
                /* 当前已经存储在内存的f[j]加载 */
                svfloat32_t fx_j_old = svld1_f32(p, fj_base + 0);
                svfloat32_t fy_j_old = svld1_f32(p, fj_base + 1);
                svfloat32_t fz_j_old = svld1_f32(p, fj_base + 2);
                
                /* 累加 */
                fx_j = svadd_f32_x(p, fx_j_old, fx_j);
                fy_j = svadd_f32_x(p, fy_j_old, fy_j);
                fz_j = svadd_f32_x(p, fz_j_old, fz_j);
                
                /* 写回 */
                svst1_f32(p, fj_base + 0, fx_j);
                svst1_f32(p, fj_base + 1, fy_j);
                svst1_f32(p, fj_base + 2, fz_j);
            }
        }
        
        /* 将累加的力写回i原子 */
        f[gi * 3 + 0] += svaddv_f32(all_p, fx_acc);
        f[gi * 3 + 1] += svaddv_f32(all_p, fy_acc);
        f[gi * 3 + 2] += svaddv_f32(all_p, fz_acc);
    }
}

/*
 * 单个Tile的融合计算
 */
void gmx_pto_nonbonded_compute_tile(gmx_pto_nonbonded_context_t *context,
                                    gmx_pto_atom_data_t *atom_data,
                                    int tile_idx) {
    if (context == NULL || atom_data == NULL || tile_idx >= context->num_tiles) {
        return;
    }
    
    gmx_pto_tile_t *tile = &context->tiles[tile_idx];
    
    /* 如果使用SME，Tile坐标已经加载到Tile寄存器 */
    /* 在本基础版本中，我们直接从内存访问 */
    /* SME优化版本会使用gmx_pto_sme_load_coords提前加载 */
    
    /* 找到所有包含此Tile的邻居对并计算 */
    for (int p = 0; p < context->num_neighbor_pairs; p++) {
        gmx_pto_neighbor_pair_t *pair = &context->neighbor_pairs[p];
        
        if (pair->tile_i == tile_idx || pair->tile_j == tile_idx) {
            gmx_pto_tile_t *tile_i = &context->tiles[pair->tile_i];
            gmx_pto_tile_t *tile_j = &context->tiles[pair->tile_j];
            
            /* 计算相互作用 */
            gmx_pto_sve_compute_pair(context, atom_data, tile_i, tile_j);
        }
    }
    
    tile->forces_computed = true;
}

/*
 * 全融合非键计算 - 整个流程融合为单个函数调用
 * 
 * 这就是PTO全流程算子融合:
 * - 原来需要多次函数调用和多次内存读写
 * - 现在一次完成，中间结果保存在寄存器
 */
int gmx_pto_nonbonded_compute_fused(gmx_pto_nonbonded_context_t *context,
                                     gmx_pto_atom_data_t *atom_data) {
    if (context == NULL || atom_data == NULL || context->tiles == NULL) {
        return -1;
    }
    
    if (context->num_neighbor_pairs == 0) {
        /* 需要先调用gmx_pto_build_neighbor_pairs */
        return -2;
    }
    
    /* 重置计算标志 */
    for (int t = 0; t < context->num_tiles; t++) {
        context->tiles[t].forces_computed = false;
    }
    
    /* 并行遍历所有Tile计算
     * 在实际GROMACS集成中，这里会用OpenMP并行
     * 每个CPU核心负责一个或多个Tile
     */
    #pragma omp parallel for
    for (int t = 0; t < context->num_tiles; t++) {
        gmx_pto_nonbonded_compute_tile(context, atom_data, t);
    }
    
    return 0;
}
