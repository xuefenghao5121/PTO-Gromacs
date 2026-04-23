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
 * PTO Tile: 编译时固定大小的2D数据块
 *
 * 使用模板参数固定 rows/cols, 使编译器能在编译时确定循环次数,
 * 实现完整的 SVE 向量化、循环展开和软件流水线。
 *
 * 对比 v6.0 (运行时 Tile2D):
 *   - v6.0: for(int i=0; i<tile.cols; i++) → 编译器不知道cols=8
 *   - v6.1: for(int i=0; i<COLS; i++) → 编译器知道COLS=8, 完全展开
 * ==================================================================== */
template<int ROWS, int COLS>
struct TileFixed {
    alignas(64) float data[ROWS * COLS];
    int valid_cols; /* 运行时有效列数 (≤ COLS) */

    TileFixed() : valid_cols(COLS) {}
    void SetValidCols(int vc) { valid_cols = vc; }

    /* 编译时大小 */
    static constexpr int rows() { return ROWS; }
    static constexpr int cols() { return COLS; }
    static constexpr int size() { return ROWS * COLS; }
};

/* ====================================================================
 * PTO GlobalTensor: 全局内存视图
 * ==================================================================== */
struct GlobalTensor1D {
    const float *data;
    int stride;

    GlobalTensor1D() : data(nullptr), stride(1) {}
    GlobalTensor1D(const float *ptr, int s = 1) : data(ptr), stride(s) {}
};

/* ====================================================================
 * PTO 算子: TLOAD — 全局→Tile (编译时展开版)
 * ==================================================================== */
template<int R, int C>
inline __attribute__((always_inline)) void TLOAD(TileFixed<R,C> &dst, const GlobalTensor1D &src) {
    if (src.stride == 1) {
        PTO_CPU_VECTORIZE_LOOP
        for (int i = 0; i < C; i++) dst.data[i] = src.data[i];
    } else {
        PTO_CPU_VECTORIZE_LOOP
        for (int i = 0; i < C; i++) dst.data[i] = src.data[i * src.stride];
    }
}

/* ====================================================================
 * PTO 算子: TFILL — 填充常量 (编译时展开)
 * ==================================================================== */
template<int R, int C>
inline __attribute__((always_inline)) void TFILL(TileFixed<R,C> &dst, float value) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) dst.data[i] = value;
}

/* ====================================================================
 * PTO 算子: TMUL — 逐元素乘法 (编译时展开)
 * ==================================================================== */
template<int R, int C>
inline __attribute__((always_inline)) void TMUL(TileFixed<R,C> &dst,
    const TileFixed<R,C> &src0, const TileFixed<R,C> &src1) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) dst.data[i] = src0.data[i] * src1.data[i];
}

/* ====================================================================
 * PTO 算子: TADD — 逐元素加法 (编译时展开)
 * ==================================================================== */
template<int R, int C>
inline __attribute__((always_inline)) void TADD(TileFixed<R,C> &dst,
    const TileFixed<R,C> &src0, const TileFixed<R,C> &src1) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) dst.data[i] = src0.data[i] + src1.data[i];
}

/* ====================================================================
 * PTO 算子: TSUB — 逐元素减法 (编译时展开)
 * ==================================================================== */
template<int R, int C>
inline __attribute__((always_inline)) void TSUB(TileFixed<R,C> &dst,
    const TileFixed<R,C> &src0, const TileFixed<R,C> &src1) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) dst.data[i] = src0.data[i] - src1.data[i];
}

/* ====================================================================
 * PTO 算子: TDIV — 逐元素除法 (编译时展开)
 * ==================================================================== */
template<int R, int C>
inline __attribute__((always_inline)) void TDIV(TileFixed<R,C> &dst,
    const TileFixed<R,C> &src0, const TileFixed<R,C> &src1) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) {
        dst.data[i] = (src1.data[i] != 0.0f) ? src0.data[i] / src1.data[i] : 0.0f;
    }
}

/* ====================================================================
 * PTO 算子: TPBC — PBC 最小镜像 (编译时展开)
 * ==================================================================== */
template<int R, int C>
inline __attribute__((always_inline)) void TPBC(TileFixed<R,C> &dx, float box, float inv_box) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) dx.data[i] -= box * rintf(dx.data[i] * inv_box);
}

/* ====================================================================
 * PTO 算子: TCONDITIONAL_INV — 条件逆 (编译时展开)
 * ==================================================================== */
template<int R, int C>
inline __attribute__((always_inline)) void TCONDITIONAL_INV(TileFixed<R,C> &ir,
    const TileFixed<R,C> &rsq, float csq, float eps = 1e-8f) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) {
        ir.data[i] = (rsq.data[i] < csq && rsq.data[i] > eps) ? 1.0f / rsq.data[i] : 0.0f;
    }
}

/* ====================================================================
 * PTO 算子: TLJ_FORCE — LJ 力融合算子 (编译时展开)
 * ==================================================================== */
template<int R, int C>
struct LJParamsT {
    float sigma_sq;
    float epsilon;
    float cutoff_sq;
    float min_rsq;
};

template<int R, int C>
inline __attribute__((always_inline)) void TLJ_FORCE(
    TileFixed<R,C> &fx, TileFixed<R,C> &fy, TileFixed<R,C> &fz,
    const TileFixed<R,C> &dx, const TileFixed<R,C> &dy, const TileFixed<R,C> &dz,
    const TileFixed<R,C> &rsq, const LJParamsT<R,C> &params,
    TileFixed<R,C> &ir, TileFixed<R,C> &s2, TileFixed<R,C> &s6, TileFixed<R,C> &s12,
    TileFixed<R,C> &fr, TileFixed<R,C> &tmp, TileFixed<R,C> &ones) {

    TCONDITIONAL_INV(ir, rsq, params.cutoff_sq, params.min_rsq);
    TFILL(ones, params.sigma_sq);
    TMUL(s2, ones, ir);
    TMUL(tmp, s2, s2);
    TMUL(s6, tmp, s2);
    TMUL(s12, s6, s6);
    TFILL(ones, 2.0f);
    TMUL(tmp, ones, s12);
    TSUB(tmp, tmp, s6);
    TMUL(fr, tmp, ir);
    TFILL(ones, 24.0f * params.epsilon);
    TMUL(fr, fr, ones);
    TMUL(fx, fr, dx);
    TMUL(fy, fr, dy);
    TMUL(fz, fr, dz);
}

/* ====================================================================
 * 运行时 Tile2D (向后兼容, 用于 dynamic 场景)
 * ==================================================================== */

} // namespace cpu
} // namespace pto

#endif /* PTO_GROMACS_CORE_HPP */
