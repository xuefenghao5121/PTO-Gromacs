/**
 * PTO-GROMACS v6 — PTO-ISA 核心算子实现
 *
 * 直接提取 PTO-ISA CPU 模拟层的核心算子:
 * - TileShape2D: PTO Tile 形状描述
 * - PTO_CPU_VECTORIZE_LOOP: 编译器向量化提示
 * - TMUL, TSUB, TADD, TDIV: 逐元素算子 (从 PTO-ISA CPU 实现提取)
 * - TLOAD, TSTORE: 数据搬运
 * - TROWTSUM: 行归约
 *
 * 这些算子直接来自 pto-isa/include/pto/cpu/*.hpp,
 * 但去掉了 NPU 相关的宏和头文件依赖,
 * 使其在 GCC 12 + C++17 + ARM SVE 上可编译。
 *
 * 参考: third_party/pto-isa/include/pto/cpu/{TMul,TSub,TAdd,TDiv,TRowSum}.hpp
 */

#ifndef PTO_GROMACS_CORE_HPP
#define PTO_GROMACS_CORE_HPP

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <omp.h>

/* ====================================================================
 * PTO 编译器向量化提示 (from pto/cpu/parallel.hpp)
 * ==================================================================== */
#if defined(__clang__)
#define PTO_CPU_PRAGMA(X) _Pragma(#X)
#define PTO_CPU_VECTORIZE_LOOP PTO_CPU_PRAGMA(clang loop vectorize(enable) interleave(enable))
#elif defined(__GNUC__)
#define PTO_CPU_PRAGMA(X) _Pragma(#X)
#define PTO_CPU_VECTORIZE_LOOP PTO_CPU_PRAGMA(GCC ivdep)
#else
#define PTO_CPU_VECTORIZE_LOOP
#endif

namespace pto {
namespace cpu {

/* ====================================================================
 * PTO Tile: 1D/2D 数据块 (from pto/common/pto_tile.hpp)
 *
 * 在 CPU 模拟层, Tile 就是连续的 float 数组,
 * 行优先布局, 对齐到缓存行。
 *
 * NPU 上的 Tile 映射到 UB (Unified Buffer),
 * 在 CPU 上就是栈/堆上的数组。
 * ==================================================================== */
struct Tile2D {
    float *data;
    int rows;
    int cols;
    int stride; /* 行步长 (= cols for row-major, compact) */

    Tile2D() : data(nullptr), rows(0), cols(0), stride(0) {}
    Tile2D(float *buf, int r, int c) : data(buf), rows(r), cols(c), stride(c) {}

    /* 动态设置有效行列 (PTO 的 valid row/col mask) */
    void SetValidShape(int r, int c) { rows = r; cols = c; stride = c; }

    float& operator()(int r, int c) { return data[r * stride + c]; }
    const float& operator()(int r, int c) const { return data[r * stride + c]; }
};

/* ====================================================================
 * PTO GlobalTensor: 全局内存视图 (from pto/common/pto_tile.hpp)
 *
 * 在 CPU 模拟层, GlobalTensor 就是一个指针 + 步长,
 * 可以指向 SoA 坐标数组中的任意位置。
 * ==================================================================== */
struct GlobalTensor1D {
    const float *data;
    int stride; /* 元素间距 (1 for contiguous, 3 for AoS xyz) */

    GlobalTensor1D() : data(nullptr), stride(1) {}
    GlobalTensor1D(const float *ptr, int s = 1) : data(ptr), stride(s) {}
};

/* ====================================================================
 * PTO 算子: TLOAD — 全局→Tile (from pto/cpu/TLoad.hpp)
 *
 * 在 CPU 模拟层, TLOAD 就是带步幅的连续拷贝。
 * 当 stride=1 时, 等价于 SVE 的 svld1 (连续加载)。
 * ==================================================================== */
inline __attribute__((always_inline)) void TLOAD(Tile2D &dst, const GlobalTensor1D &src) {
    int n = dst.rows * dst.cols;
    if (src.stride == 1) {
        /* 连续路径: 等价于 SVE svld1 */
        PTO_CPU_VECTORIZE_LOOP
        for (int i = 0; i < n; i++) {
            dst.data[i] = src.data[i];
        }
    } else {
        /* 带步幅路径: 等价于 SVE scatter/gather */
        PTO_CPU_VECTORIZE_LOOP
        for (int i = 0; i < n; i++) {
            dst.data[i] = src.data[i * src.stride];
        }
    }
}

/* ====================================================================
 * PTO 算子: TSTORE — Tile→全局 (from pto/cpu/TStore.hpp)
 * ==================================================================== */
inline __attribute__((always_inline)) void TSTORE(float *dst, const Tile2D &src, int stride = 1) {
    int n = src.rows * src.cols;
    if (stride == 1) {
        PTO_CPU_VECTORIZE_LOOP
        for (int i = 0; i < n; i++) dst[i] = src.data[i];
    } else {
        PTO_CPU_VECTORIZE_LOOP
        for (int i = 0; i < n; i++) dst[i * stride] = src.data[i];
    }
}

/* ====================================================================
 * PTO 算子: TMUL — 逐元素乘法 (from pto/cpu/TMul.hpp)
 * TMUL(dst, src0, src1): dst[i] = src0[i] * src1[i]
 * ==================================================================== */
inline __attribute__((always_inline)) void TMUL(Tile2D &dst, const Tile2D &src0, const Tile2D &src1) {
    int n = dst.rows * dst.cols;
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < n; i++) dst.data[i] = src0.data[i] * src1.data[i];
}

/* ====================================================================
 * PTO 算子: TADD — 逐元素加法 (from pto/cpu/TAdd.hpp)
 * TADD(dst, src0, src1): dst[i] = src0[i] + src1[i]
 * ==================================================================== */
inline __attribute__((always_inline)) void TADD(Tile2D &dst, const Tile2D &src0, const Tile2D &src1) {
    int n = dst.rows * dst.cols;
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < n; i++) dst.data[i] = src0.data[i] + src1.data[i];
}

/* ====================================================================
 * PTO 算子: TSUB — 逐元素减法 (from pto/cpu/TSub.hpp)
 * TSUB(dst, src0, src1): dst[i] = src0[i] - src1[i]
 * ==================================================================== */
inline __attribute__((always_inline)) void TSUB(Tile2D &dst, const Tile2D &src0, const Tile2D &src1) {
    int n = dst.rows * dst.cols;
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < n; i++) dst.data[i] = src0.data[i] - src1.data[i];
}

/* ====================================================================
 * PTO 算子: TDIV — 逐元素除法 (from pto/cpu/TDiv.hpp)
 * TDIV(dst, src0, src1): dst[i] = src0[i] / src1[i]
 * ==================================================================== */
inline __attribute__((always_inline)) void TDIV(Tile2D &dst, const Tile2D &src0, const Tile2D &src1) {
    int n = dst.rows * dst.cols;
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < n; i++) {
        dst.data[i] = (src1.data[i] != 0.0f) ? src0.data[i] / src1.data[i] : 0.0f;
    }
}

/* ====================================================================
 * PTO 算子: TFILL — 填充常量 (from pto/cpu/TAssign.hpp 概念)
 * TFILL(tile, value): tile[i] = value
 * ==================================================================== */
inline __attribute__((always_inline)) void TFILL(Tile2D &dst, float value) {
    int n = dst.rows * dst.cols;
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < n; i++) dst.data[i] = value;
}

/* ====================================================================
 * PTO 算子: TROWTSUM — 行求和 (from pto/cpu/TRowSum.hpp)
 * TROWTSUM(dst, src): dst[r] = sum(src[r,:])
 * ==================================================================== */
inline __attribute__((always_inline)) void TROWTSUM(float *dst_sum, const Tile2D &src) {
    for (int r = 0; r < src.rows; r++) {
        float sum = 0.0f;
        PTO_CPU_VECTORIZE_LOOP
        for (int c = 0; c < src.cols; c++) {
            sum += src.data[r * src.stride + c];
        }
        dst_sum[r] = sum;
    }
}

/* ====================================================================
 * PTO 算子: TSCATTER — 按索引写回 (from pto/cpu/TScatter.hpp)
 * TSCATTER(dst, indexes, src): dst[indexes[i]] += src[i]
 * ==================================================================== */
inline __attribute__((always_inline)) void TSCATTER(float *dst, const int *indexes, const Tile2D &src, float sign = -1.0f) {
    int n = src.rows * src.cols;
    for (int i = 0; i < n; i++) {
        if (indexes[i] >= 0) {
            dst[indexes[i]] += sign * src.data[i];
        }
    }
}

/* ====================================================================
 * PTO 算子: TPBC — PBC 最小镜像 (PTO-GROMACS 自定义)
 * TPBC(dx, box, inv_box): dx -= box * round(dx * inv_box)
 *
 * 在 PTO 范式中, PBC 是: TMUL→round→TMUL→TSUB
 * 但 round() 不是 PTO 内建算子, 在 CPU 层用标量处理。
 * ==================================================================== */
inline __attribute__((always_inline)) void TPBC(Tile2D &dx, float box, float inv_box) {
    int n = dx.rows * dx.cols;
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < n; i++) {
        dx.data[i] -= box * rintf(dx.data[i] * inv_box);
    }
}

/* ====================================================================
 * PTO 算子: TCONDITIONAL_INV — 条件逆 (PTO-GROMACS 自定义)
 * ir[i] = (rsq[i] < csq && rsq[i] > eps) ? 1/rsq[i] : 0
 *
 * 等价于 PTO 谓词执行: active = (rsq < csq) & (rsq > eps)
 *                       TDIV(ir, ones, rsq) under predicate
 * ==================================================================== */
inline __attribute__((always_inline)) void TCONDITIONAL_INV(Tile2D &ir, const Tile2D &rsq, float csq, float eps = 1e-8f) {
    int n = ir.rows * ir.cols;
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < n; i++) {
        ir.data[i] = (rsq.data[i] < csq && rsq.data[i] > eps) ? 1.0f / rsq.data[i] : 0.0f;
    }
}

/* ====================================================================
 * PTO 算子: TLJ_FORCE — LJ 力计算 (PTO-GROMACS 自定义融合算子)
 *
 * 输入: dx, dy, dz (距离分量), rsq (距离平方), csq (截断)
 * 输出: fx, fy, fz (力分量)
 * 中间: ir, s2, s6, s12, fr (全部在 Tile 中, 不写回全局内存)
 *
 * 算子融合链:
 *   TDIV(ir, 1, rsq) → TMUL(s2, ssq, ir) → TMUL(s6, s2, s2) → TMUL(s6, s6, s2)
 *   → TMUL(s12, s6, s6) → TMUL(tmp, 2, s12) → TSUB(tmp, tmp, s6)
 *   → TMUL(fr, tmp, ir) → TMUL(fr, fr, 24eps) → TMUL(fx, fr, dx) ...
 *
 * 参数:
 *   sigma_sq = 0.09 (σ² = 0.3²)
 *   epsilon = 0.5
 *   24*epsilon = 12.0
 * ==================================================================== */
struct LJParams {
    float sigma_sq;    /* σ² = 0.09 for LJ */
    float epsilon;     /* ε = 0.5 for LJ */
    float cutoff_sq;   /* r_c² */
    float min_rsq;     /* 1e-8 避免除零 */
};

inline __attribute__((always_inline)) void TLJ_FORCE(Tile2D &fx, Tile2D &fy, Tile2D &fz,
                       const Tile2D &dx, const Tile2D &dy, const Tile2D &dz,
                       const Tile2D &rsq, const LJParams &params,
                       /* 临时 Tile (由调用方分配, 避免函数内分配) */
                       Tile2D &ir, Tile2D &s2, Tile2D &s6, Tile2D &s12,
                       Tile2D &fr, Tile2D &tmp, Tile2D &ones) {
    /* Step 1: ir = 1/rsq (条件) */
    TCONDITIONAL_INV(ir, rsq, params.cutoff_sq, params.min_rsq);

    /* Step 2: s2 = sigma_sq * ir */
    TFILL(ones, params.sigma_sq);
    TMUL(s2, ones, ir);

    /* Step 3: s6 = s2 * s2 * s2 */
    TMUL(tmp, s2, s2);    /* tmp = s2² */
    TMUL(s6, tmp, s2);    /* s6 = s2³ */

    /* Step 4: s12 = s6 * s6 */
    TMUL(s12, s6, s6);

    /* Step 5: fr = 24*eps * (2*s12 - s6) * ir */
    TFILL(ones, 2.0f);
    TMUL(tmp, ones, s12);   /* tmp = 2*s12 */
    TSUB(tmp, tmp, s6);     /* tmp = 2*s12 - s6 */
    TMUL(fr, tmp, ir);      /* fr = (2*s12-s6) * ir */
    TFILL(ones, 24.0f * params.epsilon);
    TMUL(fr, fr, ones);     /* fr = 24*eps*(2*s12-s6)*ir */

    /* Step 6: fx=fr*dx, fy=fr*dy, fz=fr*dz */
    TMUL(fx, fr, dx);
    TMUL(fy, fr, dy);
    TMUL(fz, fr, dz);
}

} // namespace cpu
} // namespace pto

#endif /* PTO_GROMACS_CORE_HPP */
