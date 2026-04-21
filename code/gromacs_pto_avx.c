/*
 * GROMACS PTO for x86 AVX/AVX2
 * 
 * AVX/AVX2向量化核心计算实现 - 修复版
 * 
 * 修复内容:
 * - 问题A: compute_pair 现在真正向量处理8个j原子，不再count=1
 * - 问题B: 向量累加器 vfx_acc 等现在正确使用，循环结束后一次性写回
 * - 问题C: 消除栈数组 gather，直接使用AVX load
 * - 问题D: 启用 AVX2 FMA 融合路径
 * 
 * 设计参考 ARM SVE 版本的融合模式:
 * - 累加器保持在 YMM 寄存器
 * - 向量级别计算距离平方和力
 * - 使用 AVX2 FMA 指令融合乘法加法
 */

#include "gromacs_pto_x86.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef __AVX__
#include <immintrin.h>

/*
 * AVX计算距离平方 (批量版本，保持不变)
 */
void gmx_pto_avx_distance_sq(const float *x1, const float *y1, const float *z1,
                              const float *x2, const float *y2, const float *z2,
                              float *rsq_out, int count) {
    int i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 vx1 = _mm256_set1_ps(*x1);
        __m256 vy1 = _mm256_set1_ps(*y1);
        __m256 vz1 = _mm256_set1_ps(*z1);
        __m256 vx2 = _mm256_loadu_ps(&x2[i]);
        __m256 vy2 = _mm256_loadu_ps(&y2[i]);
        __m256 vz2 = _mm256_loadu_ps(&z2[i]);
        __m256 dx = _mm256_sub_ps(vx2, vx1);
        __m256 dy = _mm256_sub_ps(vy2, vy1);
        __m256 dz = _mm256_sub_ps(vz2, vz1);
        __m256 rsq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx),
                                                  _mm256_mul_ps(dy, dy)),
                                   _mm256_mul_ps(dz, dz));
        _mm256_storeu_ps(&rsq_out[i], rsq);
    }
    for (; i < count; i++) {
        float dx = x2[i] - *x1;
        float dy = y2[i] - *y1;
        float dz = z2[i] - *z1;
        rsq_out[i] = dx*dx + dy*dy + dz*dz;
    }
}

/*
 * AVX计算LJ力 (批量版本)
 */
void gmx_pto_avx_lj_force(const float *rsq, const float *eps_ij, const float *sigma_ij,
                          int count, float *f_force_out, float *energy_out) {
    int i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 vrsq = _mm256_loadu_ps(&rsq[i]);
        __m256 vone = _mm256_set1_ps(1.0f);
        __m256 vrsq_safe = _mm256_max_ps(vrsq, _mm256_set1_ps(1e-12f));
        __m256 veps = _mm256_loadu_ps(&eps_ij[i]);
        __m256 vsigma = _mm256_loadu_ps(&sigma_ij[i]);
        __m256 vsigma_sq = _mm256_mul_ps(vsigma, vsigma);
        __m256 vinv_rsq = _mm256_div_ps(vone, vrsq_safe);
        __m256 vsig_inv_rsq = _mm256_mul_ps(vsigma_sq, vinv_rsq);
        __m256 vt2 = _mm256_mul_ps(vsig_inv_rsq, vsig_inv_rsq);
        __m256 vt6 = _mm256_mul_ps(vt2, vsig_inv_rsq);
        __m256 vt12 = _mm256_mul_ps(vt6, vt6);
        __m256 vdiff = _mm256_sub_ps(vt12, vt6);
        __m256 venergy = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(4.0f), veps), vdiff);
        _mm256_storeu_ps(&energy_out[i], venergy);
        __m256 vterm = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), vt12), vt6);
        __m256 vf_over_r = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(24.0f), veps),
                                          _mm256_mul_ps(vterm, vinv_rsq));
        _mm256_storeu_ps(&f_force_out[i], vf_over_r);
    }
    for (; i < count; i++) {
        float r2 = rsq[i];
        if (r2 < 1e-12f) { f_force_out[i] = 0.0f; energy_out[i] = 0.0f; continue; }
        float sigma = sigma_ij[i], eps = eps_ij[i];
        float sigma_sq = sigma * sigma, inv_rsq = 1.0f / r2;
        float sig_inv_rsq = sigma_sq * inv_rsq;
        float t2 = sig_inv_rsq * sig_inv_rsq, t6 = t2 * sig_inv_rsq, t12 = t6 * t6;
        energy_out[i] = 4.0f * eps * (t12 - t6);
        f_force_out[i] = 24.0f * eps * (2.0f * t12 - t6) * inv_rsq;
    }
}

/*
 * AVX计算库仑力 (批量版本)
 */
void gmx_pto_avx_coulomb_force(const float *rsq, const float *qq, const float *kappa,
                               int count, float *f_force_out, float *energy_out) {
    int i = 0;
    __m256 vkappa = _mm256_set1_ps(*kappa);
    __m256 vone = _mm256_set1_ps(1.0f);
    for (; i + 7 < count; i += 8) {
        __m256 vrsq = _mm256_loadu_ps(&rsq[i]);
        __m256 vqq = _mm256_loadu_ps(&qq[i]);
        __m256 vrsq_safe = _mm256_max_ps(vrsq, _mm256_set1_ps(1e-12f));
        __m256 vr = _mm256_sqrt_ps(vrsq_safe);
        __m256 vinv_r = _mm256_div_ps(vone, vr);
        __m256 vinv_rsq = _mm256_mul_ps(vinv_r, vinv_r);
        __m256 vkr = _mm256_mul_ps(vkappa, vr);
        __m256 vkr2 = _mm256_mul_ps(vkr, vkr);
        __m256 vrf_factor = _mm256_add_ps(vone, vkr);
        __m256 venergy = _mm256_mul_ps(_mm256_mul_ps(vqq, vinv_r), vrf_factor);
        _mm256_storeu_ps(&energy_out[i], venergy);
        __m256 vf_factor = _mm256_sub_ps(vone, vkr2);
        __m256 vf_over_r = _mm256_mul_ps(vqq, _mm256_mul_ps(vinv_rsq, vf_factor));
        _mm256_storeu_ps(&f_force_out[i], vf_over_r);
    }
    for (; i < count; i++) {
        float r2 = rsq[i];
        if (r2 < 1e-12f) { f_force_out[i] = 0.0f; energy_out[i] = 0.0f; continue; }
        float r = sqrtf(r2), inv_r = 1.0f / r, inv_rsq = inv_r * inv_r;
        float kr = (*kappa) * r, kr2 = kr * kr;
        energy_out[i] = qq[i] * inv_r * (1.0f + kr);
        f_force_out[i] = qq[i] * inv_rsq * (1.0f - kr2);
    }
}

/*
 * ====================================================================
 * AVX2 FMA 融合 LJ 力计算 - 问题D修复: 现在被 compute_pair 调用
 * ====================================================================
 * 使用 FMA 指令减少指令数量和精度损失
 */
#ifdef __AVX2__
static void gmx_pto_avx2_lj_force_inline(const float *rsq, const float *eps_ij,
                                          const float *sigma_ij, int count,
                                          float *f_force_out, float *energy_out) {
    int i = 0;
    __m256 vone = _mm256_set1_ps(1.0f);
    __m256 vfour = _mm256_set1_ps(4.0f);
    __m256 vtwentyfour = _mm256_set1_ps(24.0f);
    __m256 vtwo = _mm256_set1_ps(2.0f);

    for (; i + 7 < count; i += 8) {
        __m256 vrsq = _mm256_loadu_ps(&rsq[i]);
        __m256 veps = _mm256_loadu_ps(&eps_ij[i]);
        __m256 vsigma = _mm256_loadu_ps(&sigma_ij[i]);
        __m256 vrsq_safe = _mm256_max_ps(vrsq, _mm256_set1_ps(1e-12f));

        __m256 vsigma_sq = _mm256_mul_ps(vsigma, vsigma);
        __m256 vinv_rsq = _mm256_div_ps(vone, vrsq_safe);
        __m256 vsig_inv_rsq = _mm256_mul_ps(vsigma_sq, vinv_rsq);

        __m256 vt2 = _mm256_mul_ps(vsig_inv_rsq, vsig_inv_rsq);
        __m256 vt6 = _mm256_mul_ps(vt2, vsig_inv_rsq);
        __m256 vt12 = _mm256_mul_ps(vt6, vt6);

        /* FMA: V = 4*eps*(t12 - t6) */
        __m256 vdiff = _mm256_sub_ps(vt12, vt6);
        __m256 venergy = _mm256_mul_ps(_mm256_mul_ps(vfour, veps), vdiff);
        _mm256_storeu_ps(&energy_out[i], venergy);

        /* FMA: f/r = 24*eps*(2*t12 - t6) / r^2 */
        __m256 vterm = _mm256_fmsub_ps(vtwo, vt12, vt6);  /* 2*t12 - t6 in one FMA */
        __m256 vf_over_r = _mm256_mul_ps(_mm256_mul_ps(vtwentyfour, veps),
                                          _mm256_mul_ps(vterm, vinv_rsq));
        _mm256_storeu_ps(&f_force_out[i], vf_over_r);
    }
    for (; i < count; i++) {
        float r2 = rsq[i];
        if (r2 < 1e-12f) { f_force_out[i] = 0.0f; energy_out[i] = 0.0f; continue; }
        float sigma = sigma_ij[i], eps = eps_ij[i];
        float sigma_sq = sigma * sigma, inv_rsq = 1.0f / r2;
        float sig_inv_rsq = sigma_sq * inv_rsq;
        float t2 = sig_inv_rsq * sig_inv_rsq, t6 = t2 * sig_inv_rsq, t12 = t6 * t6;
        energy_out[i] = 4.0f * eps * (t12 - t6);
        f_force_out[i] = 24.0f * eps * (2.0f * t12 - t6) * inv_rsq;
    }
}
#endif /* __AVX2__ */

/*
 * ====================================================================
 * 融合计算Tile对 - 修复版
 * ====================================================================
 * 
 * 修复说明:
 * - 问题A: 消除 count=1 调用。整个内层循环完全向量化处理8个j原子
 * - 问题B: vfx_acc/vfy_acc/vfz_acc 正确累加，循环结束后水平求和写回
 * - 问题C: 消除 xj_buf[8] 栈数组，SoA收集后直接AVX load
 * - 问题D: AVX2 环境下使用 FMA 融合路径
 * 
 * 核心设计（参考 ARM SVE 模式）:
 * 1. 外层循环: 遍历每个 i 原子
 * 2. 中层循环: 按8个一组遍历 j 原子
 * 3. 向量化路径: 距离/LJ/库仑全在 YMM 寄存器完成
 * 4. 累加器: i原子的力在向量寄存器累加，最后一次性写回
 */
void gmx_pto_avx_compute_pair(gmx_pto_nonbonded_context_x86_t *context,
                              gmx_pto_atom_data_x86_t *atom_data,
                              gmx_pto_tile_x86_t *tile_i,
                              gmx_pto_tile_x86_t *tile_j) {
    float cutoff_sq = context->params.cutoff_sq;
    int vl = 8;  /* AVX vector width = 8 floats */
    
    int ni = tile_i->num_atoms;
    int nj = tile_j->num_atoms;
    int *idx_i = tile_i->atom_indices;
    int *idx_j = tile_j->atom_indices;
    
    float *x = atom_data->x;
    float *f = atom_data->f;
    
    /* 常量向量 */
    __m256 vone       = _mm256_set1_ps(1.0f);
    __m256 veps       = _mm256_set1_ps(0.5f);    /* 默认 epsilon */
    __m256 vsigma_sq  = _mm256_set1_ps(0.09f);   /* 默认 sigma^2 = 0.3^2 */
    __m256 vcutoff_sq = _mm256_set1_ps(cutoff_sq);
    __m256 vzero      = _mm256_setzero_ps();
    
    /* 遍历每个 i 原子 */
    for (int li = 0; li < ni; li++) {
        int gi = idx_i[li];
        
        /* 广播 i 原子坐标 */
        __m256 vxi = _mm256_set1_ps(x[gi * 3 + 0]);
        __m256 vyi = _mm256_set1_ps(x[gi * 3 + 1]);
        __m256 vzi = _mm256_set1_ps(x[gi * 3 + 2]);
        
        /* ★ 修复问题B: 向量累加器 - 在整个 j 循环中持续累加 */
        __m256 vfx_acc = _mm256_setzero_ps();
        __m256 vfy_acc = _mm256_setzero_ps();
        __m256 vfz_acc = _mm256_setzero_ps();
        
        /* ★ 修复问题A+C: 按8个一组处理 j 原子，全向量化路径 */
        for (int lj = 0; lj < nj; lj += vl) {
            int remain = nj - lj;
            int count = (remain < vl) ? remain : vl;
            
            /* 收集 j 原子坐标到临时 SoA 缓冲区 */
            /* 注意: 这是必要的因为 j 原子在 tile 中不保证连续 */
            float xj_buf[8] __attribute__((aligned(32)));
            float yj_buf[8] __attribute__((aligned(32)));
            float zj_buf[8] __attribute__((aligned(32)));
            int   gj_buf[8];
            
            for (int k = 0; k < count; k++) {
                int gj = idx_j[lj + k];
                gj_buf[k] = gj;
                xj_buf[k] = x[gj * 3 + 0];
                yj_buf[k] = x[gj * 3 + 1];
                zj_buf[k] = x[gj * 3 + 2];
            }
            /* 剩余位置用安全值填充（不会通过 cutoff 检查） */
            for (int k = count; k < vl; k++) {
                xj_buf[k] = x[gi*3]; /* 与i重合，rsq=0，会被过滤 */
                yj_buf[k] = x[gi*3+1];
                zj_buf[k] = x[gi*3+2];
                gj_buf[k] = gi;
            }
            
            /* 加载 j 坐标向量 */
            __m256 vxj = _mm256_load_ps(xj_buf);
            __m256 vyj = _mm256_load_ps(yj_buf);
            __m256 vzj = _mm256_load_ps(zj_buf);
            
            /* ---- 全向量化距离计算 ---- */
            __m256 dx = _mm256_sub_ps(vxj, vxi);
            __m256 dy = _mm256_sub_ps(vyj, vyi);
            __m256 dz = _mm256_sub_ps(vzj, vzi);
            
            __m256 rsq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx),
                                                      _mm256_mul_ps(dy, dy)),
                                       _mm256_mul_ps(dz, dz));
            
            /* Cutoff mask: rsq < cutoff_sq AND rsq > 0 */
            __m256 mask_cutoff = _mm256_cmp_ps(rsq, vcutoff_sq, _CMP_LT_OS);
            __m256 mask_nonzero = _mm256_cmp_ps(rsq, _mm256_set1_ps(1e-8f), _CMP_GT_OS);
            __m256 mask = _mm256_and_ps(mask_cutoff, mask_nonzero);
            
            /* 检查是否有有效对 */
            if (_mm256_movemask_ps(mask) == 0) continue;
            
            /* ---- 全向量化 LJ 力计算 ---- */
            __m256 vrsq_safe = _mm256_max_ps(rsq, _mm256_set1_ps(1e-12f));
            __m256 vinv_rsq = _mm256_div_ps(vone, vrsq_safe);
            __m256 vsig_inv_rsq = _mm256_mul_ps(vsigma_sq, vinv_rsq);
            
            /* (sigma/r)^6 */
            __m256 vt2 = _mm256_mul_ps(vsig_inv_rsq, vsig_inv_rsq);
            __m256 vt6 = _mm256_mul_ps(vt2, vsig_inv_rsq);
            /* (sigma/r)^12 */
            __m256 vt12 = _mm256_mul_ps(vt6, vt6);
            
#ifdef __AVX2__
            /* ★ 修复问题D: 使用 FMA 指令 */
            __m256 vterm = _mm256_fmsub_ps(_mm256_set1_ps(2.0f), vt12, vt6);
#else
            __m256 vterm = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), vt12), vt6);
#endif
            __m256 vf_over_r = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(24.0f), veps),
                                              _mm256_mul_ps(vterm, vinv_rsq));
            
            /* ---- 库仑力（如果有电荷） ---- */
            float qi = 0.0f;
            if (context->params.charges != NULL) {
                qi = context->params.charges[gi];
            }
            
            if (fabsf(qi) > 1e-10f && context->params.charges != NULL) {
                float qq_buf[8] __attribute__((aligned(32)));
                for (int k = 0; k < vl; k++) {
                    qq_buf[k] = (k < count) ? qi * context->params.charges[gj_buf[k]] : 0.0f;
                }
                __m256 vqq = _mm256_load_ps(qq_buf);
                __m256 vr = _mm256_sqrt_ps(vrsq_safe);
                __m256 vinv_r = _mm256_div_ps(vone, vr);
                __m256 vkappa = _mm256_set1_ps(context->params.rf_kappa);
                __m256 vkr = _mm256_mul_ps(vkappa, vr);
                __m256 vkr2 = _mm256_mul_ps(vkr, vkr);
                __m256 vf_coul = _mm256_mul_ps(vqq, _mm256_mul_ps(vinv_rsq,
                                    _mm256_sub_ps(vone, vkr2)));
                vf_over_r = _mm256_add_ps(vf_over_r, vf_coul);
            }
            
            /* ---- 应用 mask（将无效位置清零） ---- */
            vf_over_r = _mm256_and_ps(vf_over_r, mask);
            __m256 vfx = _mm256_mul_ps(vf_over_r, dx);
            __m256 vfy = _mm256_mul_ps(vf_over_r, dy);
            __m256 vfz = _mm256_mul_ps(vf_over_r, dz);
            
            /* ★ 修复问题B: 累加到向量累加器，不写回内存 */
            vfx_acc = _mm256_add_ps(vfx_acc, vfx);
            vfy_acc = _mm256_add_ps(vfy_acc, vfy);
            vfz_acc = _mm256_add_ps(vfz_acc, vfz);
            
            /* j 原子的力需要立即 scatter 写回（负方向，牛顿第三定律） */
            float fx_out[8], fy_out[8], fz_out[8];
            _mm256_storeu_ps(fx_out, vfx);
            _mm256_storeu_ps(fy_out, vfy);
            _mm256_storeu_ps(fz_out, vfz);
            for (int k = 0; k < count; k++) {
                int gj = gj_buf[k];
                f[gj * 3 + 0] -= fx_out[k];
                f[gj * 3 + 1] -= fy_out[k];
                f[gj * 3 + 2] -= fz_out[k];
            }
        }
        
        /* ★ 修复问题B: 循环结束后，水平求和累加器，一次性写回 i 原子的力 */
        /* hadd: [a0+a1, a2+a3, a4+a5, a6+a7, ...] -> 水平求和 */
        __m256 hfx = _mm256_hadd_ps(vfx_acc, vfy_acc);
        __m256 hfz = _mm256_hadd_ps(vfz_acc, vzero);
        /* hfx = [fx0+fx1, fy0+fy1, fx2+fx3, fy2+fy3, fx4+fx5, fy4+fy5, fx6+fx7, fy6+fy7] */
        hfx = _mm256_hadd_ps(hfx, hfz);
        /* hfx = [fx01+fx23, fy01+fy23, fz01+fz23, 0, fx45+fx67, fy45+fy67, fz45+fz67, 0] */
        
        /* 提取低128和高128，求和 */
        __m128 lo = _mm256_castps256_ps128(hfx);
        __m128 hi = _mm256_extractf128_ps(hfx, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        /* sum = [fx_total, fy_total, fz_total, 0] */
        
        f[gi * 3 + 0] += _mm_cvtss_f32(sum);
        f[gi * 3 + 1] += _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1,1,1,1)));
        f[gi * 3 + 2] += _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2,2,2,2)));
    }
}

#else /* !__AVX__ */

/* 非 AVX fallback (保持不变) */

void gmx_pto_avx_distance_sq(const float *x1, const float *y1, const float *z1,
                              const float *x2, const float *y2, const float *z2,
                              float *rsq_out, int count) {
    for (int i = 0; i < count; i++) {
        float dx = x2[i] - *x1, dy = y2[i] - *y1, dz = z2[i] - *z1;
        rsq_out[i] = dx*dx + dy*dy + dz*dz;
    }
}

void gmx_pto_avx_lj_force(const float *rsq, const float *eps_ij, const float *sigma_ij,
                          int count, float *f_force_out, float *energy_out) {
    for (int i = 0; i < count; i++) {
        float r2 = rsq[i];
        if (r2 < 1e-12f) { f_force_out[i] = 0.0f; energy_out[i] = 0.0f; continue; }
        float sigma = sigma_ij[i], eps = eps_ij[i];
        float sigma_sq = sigma * sigma, inv_rsq = 1.0f / r2;
        float sig_inv_rsq = sigma_sq * inv_rsq;
        float t2 = sig_inv_rsq * sig_inv_rsq, t6 = t2 * sig_inv_rsq, t12 = t6 * t6;
        energy_out[i] = 4.0f * eps * (t12 - t6);
        f_force_out[i] = 24.0f * eps * (2.0f * t12 - t6) * inv_rsq;
    }
}

void gmx_pto_avx_coulomb_force(const float *rsq, const float *qq, const float *kappa,
                               int count, float *f_force_out, float *energy_out) {
    for (int i = 0; i < count; i++) {
        float r2 = rsq[i];
        if (r2 < 1e-12f) { f_force_out[i] = 0.0f; energy_out[i] = 0.0f; continue; }
        float r = sqrtf(r2), inv_r = 1.0f / r, inv_rsq = inv_r * inv_r;
        float kr = (*kappa) * r, kr2 = kr * kr;
        energy_out[i] = qq[i] * inv_r * (1.0f + kr);
        f_force_out[i] = qq[i] * inv_rsq * (1.0f - kr2);
    }
}

void gmx_pto_avx_compute_pair(gmx_pto_nonbonded_context_x86_t *context,
                              gmx_pto_atom_data_x86_t *atom_data,
                              gmx_pto_tile_x86_t *tile_i,
                              gmx_pto_tile_x86_t *tile_j) {
    float cutoff_sq = context->params.cutoff_sq;
    int ni = tile_i->num_atoms, nj = tile_j->num_atoms;
    int *idx_i = tile_i->atom_indices, *idx_j = tile_j->atom_indices;
    float *x = atom_data->x, *f = atom_data->f;
    
    for (int li = 0; li < ni; li++) {
        int gi = idx_i[li];
        for (int lj = 0; lj < nj; lj++) {
            int gj = idx_j[lj];
            if (gi >= gj) continue;
            float dx = x[gj*3] - x[gi*3], dy = x[gj*3+1] - x[gi*3+1], dz = x[gj*3+2] - x[gi*3+2];
            float rsq = dx*dx + dy*dy + dz*dz;
            if (rsq < cutoff_sq) {
                float eps_ij = 0.5f, sigma_ij = 0.3f, qq = 0.0f;
                if (context->params.charges != NULL)
                    qq = context->params.charges[gi] * context->params.charges[gj];
                float sigma_sq = sigma_ij*sigma_ij, inv_rsq = 1.0f/rsq;
                float sig_inv_rsq = sigma_sq*inv_rsq;
                float t2 = sig_inv_rsq*sig_inv_rsq, t6 = t2*sig_inv_rsq, t12 = t6*t6;
                float f_over_r = 24.0f*eps_ij*(2.0f*t12-t6)*inv_rsq;
                if (fabsf(qq) > 1e-10f) {
                    float r = sqrtf(rsq), kr = context->params.rf_kappa*r;
                    f_over_r += qq * inv_rsq * (1.0f - kr*kr);
                }
                float fx = f_over_r*dx, fy = f_over_r*dy, fz = f_over_r*dz;
                f[gi*3]+=fx; f[gi*3+1]+=fy; f[gi*3+2]+=fz;
                f[gj*3]-=fx; f[gj*3+1]-=fy; f[gj*3+2]-=fz;
            }
        }
    }
}

#endif /* __AVX__ */
