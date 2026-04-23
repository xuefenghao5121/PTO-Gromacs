/**
 * PTO-GROMACS v7 — PTO-ISA 多后端算子实现
 *
 * 核心架构:
 *   PTO-ISA API (可移植接口)
 *       ↓  编译时选择
 *   Backend Implementation
 *       ├── ARM SVE:  svld1_f32, svmul_f32_x, svsub_f32_x ...
 *       ├── Generic:  PTO_CPU_VECTORIZE_LOOP + for loop
 *       └── (future) x86 AVX-512, NPU native
 *
 * 当 COLS == SVE_VECTOR_BITS/32 时, 每个 PTO 算子
 * 1:1 映射到一条 SVE 指令, 性能等价于手写 SVE intrinsics。
 *
 * 参考: third_party/pto-isa/include/pto/cpu/*.hpp (NPU 原始实现)
 */

#ifndef PTO_GROMACS_CORE_HPP
#define PTO_GROMACS_CORE_HPP

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <omp.h>

/* ====================================================================
 * Backend Selection
 *
 * HAVE_SVE: 使用 ARM SVE intrinsics 后端 (最优)
 * 否则:     使用 Generic CPU 后端 (编译器自定向量化)
 * ==================================================================== */
#if defined(HAVE_SVE) && defined(__ARM_FEATURE_SVE)
  #include <arm_sve.h>
  #define PTO_BACKEND_SVE 1
#else
  #define PTO_BACKEND_SVE 0
  /* Generic backend: compiler vectorization hints */
  #if defined(__clang__)
    #define PTO_CPU_PRAGMA(X) _Pragma(#X)
    #define PTO_CPU_VECTORIZE_LOOP PTO_CPU_PRAGMA(clang loop vectorize(enable) interleave(enable))
  #elif defined(__GNUC__)
    #define PTO_CPU_PRAGMA(X) _Pragma(#X)
    #define PTO_CPU_VECTORIZE_LOOP PTO_CPU_PRAGMA(GCC ivdep)
  #else
    #define PTO_CPU_VECTORIZE_LOOP
  #endif
#endif

namespace pto {
namespace cpu {

/* ====================================================================
 * PTO Tile: 编译时固定大小的数据块
 *
 * SVE 后端: Tile 大小 = SVE 向量宽度 (256-bit = 8 floats)
 *           每个 PTO 算子 = 一条 SVE 指令
 * Generic 后端: Tile 大小 = 编译时常量, 依赖循环展开
 * ==================================================================== */
template<int ROWS, int COLS>
struct TileFixed {
    alignas(64) float data[ROWS * COLS];
    int valid_cols;

    TileFixed() : valid_cols(COLS) {}
    void SetValidCols(int vc) { valid_cols = vc; }

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
 * PTO 算子参数
 * ==================================================================== */
template<int R, int C>
struct LJParamsT {
    float sigma_sq;
    float epsilon;
    float cutoff_sq;
    float min_rsq;
};

/* ====================================================================
 * ================================================================== *
 *  ARM SVE Backend                                                    *
 *  每个 PTO 算子直接映射到 SVE intrinsics                            *
 *  TileFixed<1,8> 的 data[8] = 一个 SVE 256-bit 向量                 *
 * ================================================================== *
 * ==================================================================== */
#if PTO_BACKEND_SVE

/* --- TLOAD: svld1_f32 --- */
template<int R, int C>
inline __attribute__((always_inline)) void TLOAD(TileFixed<R,C> &dst, const GlobalTensor1D &src) {
    svbool_t pg = svptrue_b32();
    if (src.stride == 1) {
        /* 连续加载: 等价于 svld1 */
        svfloat32_t v = svld1_f32(pg, src.data);
        svst1_f32(pg, dst.data, v);
    } else {
        /* 带步幅: gather */
        svint32_t idx = svindex_s32(0, src.stride);
        svfloat32_t v = svld1_gather_s32index_f32(pg, src.data, idx);
        svst1_f32(pg, dst.data, v);
    }
}

/* --- TFILL: svdup_f32 --- */
template<int R, int C>
inline __attribute__((always_inline)) void TFILL(TileFixed<R,C> &dst, float value) {
    svfloat32_t v = svdup_f32(value);
    svst1_f32(svptrue_b32(), dst.data, v);
}

/* --- TMUL: svmul_f32_x --- */
template<int R, int C>
inline __attribute__((always_inline)) void TMUL(TileFixed<R,C> &dst,
    const TileFixed<R,C> &src0, const TileFixed<R,C> &src1) {
    svbool_t pg = svptrue_b32();
    svfloat32_t a = svld1_f32(pg, src0.data);
    svfloat32_t b = svld1_f32(pg, src1.data);
    svfloat32_t r = svmul_f32_x(pg, a, b);
    svst1_f32(pg, dst.data, r);
}

/* --- TADD: svadd_f32_x --- */
template<int R, int C>
inline __attribute__((always_inline)) void TADD(TileFixed<R,C> &dst,
    const TileFixed<R,C> &src0, const TileFixed<R,C> &src1) {
    svbool_t pg = svptrue_b32();
    svfloat32_t a = svld1_f32(pg, src0.data);
    svfloat32_t b = svld1_f32(pg, src1.data);
    svfloat32_t r = svadd_f32_x(pg, a, b);
    svst1_f32(pg, dst.data, r);
}

/* --- TSUB: svsub_f32_x --- */
template<int R, int C>
inline __attribute__((always_inline)) void TSUB(TileFixed<R,C> &dst,
    const TileFixed<R,C> &src0, const TileFixed<R,C> &src1) {
    svbool_t pg = svptrue_b32();
    svfloat32_t a = svld1_f32(pg, src0.data);
    svfloat32_t b = svld1_f32(pg, src1.data);
    svfloat32_t r = svsub_f32_x(pg, a, b);
    svst1_f32(pg, dst.data, r);
}

/* --- TDIV: svdiv_f32_z (条件) --- */
template<int R, int C>
inline __attribute__((always_inline)) void TDIV(TileFixed<R,C> &dst,
    const TileFixed<R,C> &src0, const TileFixed<R,C> &src1) {
    svbool_t pg = svptrue_b32();
    svfloat32_t a = svld1_f32(pg, src0.data);
    svfloat32_t b = svld1_f32(pg, src1.data);
    svbool_t nz = svcmpne_f32(pg, b, svdup_f32(0.0f));
    svfloat32_t r = svdiv_f32_z(nz, a, b);
    svst1_f32(pg, dst.data, r);
}

/* --- TPBC: PBC 最小镜像 ---
 * dx -= box * rint(dx * inv_box)
 * SVE: svmul → svrinta → svmul → svsub
 */
template<int R, int C>
inline __attribute__((always_inline)) void TPBC(TileFixed<R,C> &dx, float box, float inv_box) {
    svbool_t pg = svptrue_b32();
    svfloat32_t v = svld1_f32(pg, dx.data);
    v = svsub_f32_x(pg, v,
        svmul_f32_x(pg, svdup_f32(box),
            svrinta_f32_x(pg, svmul_f32_x(pg, v, svdup_f32(inv_box)))));
    svst1_f32(pg, dx.data, v);
}

/* --- TCONDITIONAL_INV: 条件逆 (SVE predicate) ---
 * ir = (rsq < csq && rsq > eps) ? 1/rsq : 0
 * SVE: svcmplt + svcmpgt → svand → svdiv_f32_z
 */
template<int R, int C>
inline __attribute__((always_inline)) void TCONDITIONAL_INV(TileFixed<R,C> &ir,
    const TileFixed<R,C> &rsq, float csq, float eps = 1e-8f) {
    svbool_t pg = svptrue_b32();
    svfloat32_t r = svld1_f32(pg, rsq.data);
    svbool_t valid = svand_b_z(pg,
        svcmplt_f32(pg, r, svdup_f32(csq)),
        svcmpgt_f32(pg, r, svdup_f32(eps)));
    svfloat32_t result = svdiv_f32_z(valid, svdup_f32(1.0f), r);
    svst1_f32(pg, ir.data, result);
}

/* --- TNONBONDED_LJ: 超级融合算子 (SVE 全流水线, 零中间写回) ---
 *
 * 把整条 non-bonded LJ 计算链在 SVE 寄存器中完成:
 *   load(xj) → sub(dx) → pbc(dx) → mul/add(rsq) → lj_force → adda(fi) → st(fj)
 *
 * 输入: 全局内存 (sx, sy, sz 坐标数组)
 * 输出: 累加到 fi, 写出到 fj 数组
 * 中间: 全部在 SVE 寄存器 z0-z15 中, 不写回内存
 *
 * SVE 寄存器分配 (16 个 z 寄存器):
 *   z0:  xj, z1: yj, z2: zj     (j坐标, 输入)
 *   z3:  dx, z4: dy, z5: dz     (距离, 中间)
 *   z6:  rsq                     (距离平方)
 *   z7:  ir                      (1/rsq)
 *   z8:  s2, z9: s6, z10: s12   (LJ中间量)
 *   z11: fr                      (力标量)
 *   z12: fx, z13: fy, z14: fz   (力分量)
 *   z15: 广播常量 (xi, box, sigma_sq, eps 等)
 *   p0-p3: 谓词寄存器
 *
 * 内存访问: 3 load (xj,yj,zj) + 3 store (fx,fy,fz) = 6 次
 * 对比 v7: 17 算子 × (2 load + 1 store) = 51 次
 * 节省: 45 次内存访问 = 88% 减少
 */
struct NonBondedParams {
    float xi, yi, zi;       /* i 原子坐标 */
    float box[3];           /* 盒子尺寸 */
    float inv_box[3];        /* 盒子逆 */
    float sigma_sq;          /* LJ σ² */
    float epsilon;           /* LJ ε */
    float cutoff_sq;         /* 截断² */
    float min_rsq;           /* 最小距离² */
};

inline __attribute__((always_inline)) void TNONBONDED_LJ(
    const float *sx, const float *sy, const float *sz,  /* 全局坐标 */
    int j0, int tile_n,                                  /* j 原子范围 */
    const NonBondedParams &p,                            /* 参数 */
    float &fix, float &fiy, float &fiz,                  /* i 力累加 */
    float *lfx, float *lfy, float *lfz) {                /* j 力数组 */

    svbool_t pg_all = svptrue_b32();
    svbool_t pg = (tile_n < 8) ? svwhilelt_b32(0, tile_n) : pg_all;

    /* Step 1: 加载 j 坐标到 SVE 寄存器 */
    svfloat32_t xj = svld1_f32(pg, &sx[j0]);
    svfloat32_t yj = svld1_f32(pg, &sy[j0]);
    svfloat32_t zj = svld1_f32(pg, &sz[j0]);

    /* Step 2: dx = xi - xj (广播 xi) */
    svfloat32_t dx = svsub_f32_x(pg, svdup_f32(p.xi), xj);
    svfloat32_t dy = svsub_f32_x(pg, svdup_f32(p.yi), yj);
    svfloat32_t dz = svsub_f32_x(pg, svdup_f32(p.zi), zj);

    /* Step 3: PBC 最小镜像 */
    dx = svsub_f32_x(pg, dx,
        svmul_f32_x(pg, svdup_f32(p.box[0]),
            svrinta_f32_x(pg, svmul_f32_x(pg, dx, svdup_f32(p.inv_box[0])))));
    dy = svsub_f32_x(pg, dy,
        svmul_f32_x(pg, svdup_f32(p.box[1]),
            svrinta_f32_x(pg, svmul_f32_x(pg, dy, svdup_f32(p.inv_box[1])))));
    dz = svsub_f32_x(pg, dz,
        svmul_f32_x(pg, svdup_f32(p.box[2]),
            svrinta_f32_x(pg, svmul_f32_x(pg, dz, svdup_f32(p.inv_box[2])))));

    /* Step 4: rsq = dx² + dy² + dz² */
    svfloat32_t rsq = svadd_f32_x(pg, svadd_f32_x(pg,
        svmul_f32_x(pg, dx, dx), svmul_f32_x(pg, dy, dy)),
        svmul_f32_x(pg, dz, dz));

    /* Step 5: 谓词 - valid = (rsq < cutoff²) && (rsq > min_rsq) */
    svbool_t valid = svand_b_z(pg_all,
        svcmplt_f32(pg_all, rsq, svdup_f32(p.cutoff_sq)),
        svcmpgt_f32(pg_all, rsq, svdup_f32(p.min_rsq)));

    /* Step 6: ir = 1/rsq (under predicate) */
    svfloat32_t ir = svdiv_f32_z(valid, svdup_f32(1.0f), rsq);

    /* Step 7: s2 = sigma_sq * ir */
    svfloat32_t s2 = svmul_f32_z(valid, svdup_f32(p.sigma_sq), ir);

    /* Step 8: s6 = s2³ */
    svfloat32_t s6 = svmul_f32_z(valid, svmul_f32_z(valid, s2, s2), s2);

    /* Step 9: s12 = s6² */
    svfloat32_t s12 = svmul_f32_z(valid, s6, s6);

    /* Step 10: fr = 24*eps * (2*s12 - s6) * ir */
    svfloat32_t fr = svmul_f32_z(valid,
        svsub_f32_z(valid, svmul_f32_z(valid, svdup_f32(2.0f), s12), s6),
        ir);
    fr = svmul_f32_z(valid, fr, svdup_f32(24.0f * p.epsilon));

    /* Step 11: fx = fr * dx, fy = fr * dy, fz = fr * dz */
    svfloat32_t fx = svmul_f32_z(valid, fr, dx);
    svfloat32_t fy = svmul_f32_z(valid, fr, dy);
    svfloat32_t fz = svmul_f32_z(valid, fr, dz);

    /* Step 12: i 力累加 (svadda 横向归约, 用 valid predicate) */
    fix += svadda_f32(valid, 0.0f, fx);
    fiy += svadda_f32(valid, 0.0f, fy);
    fiz += svadda_f32(valid, 0.0f, fz);

    /* Step 13: j 力写回 (向量化 read-modify-write) */
    svfloat32_t old_lfx = svld1_f32(pg, &lfx[j0]);
    svfloat32_t old_lfy = svld1_f32(pg, &lfy[j0]);
    svfloat32_t old_lfz = svld1_f32(pg, &lfz[j0]);
    svst1_f32(pg, &lfx[j0], svsub_f32_x(pg, old_lfx, fx));
    svst1_f32(pg, &lfy[j0], svsub_f32_x(pg, old_lfy, fy));
    svst1_f32(pg, &lfz[j0], svsub_f32_x(pg, old_lfz, fz));
}

/* --- TLJ_FORCE: LJ力融合算子 (SVE 全流水线) ---
 *
 * 整个 LJ 力计算在一个函数内完成, SVE 向量全程在寄存器中,
 * 不写回内存 → 真正的算子融合。
 *
 * SVE 寄存器分配 (最多需要 ~14 个 z 寄存器, SVE 有 32 个):
 *   z0: dx, z1: dy, z2: dz     (距离, 输入)
 *   z3: rsq                      (距离平方)
 *   z4: ir                       (1/rsq)
 *   z5: s2, z6: s6, z7: s12     (LJ 中间量)
 *   z8: fr                       (力标量)
 *   z9: fx, z10: fy, z11: fz    (力分量, 输出)
 *   z12-z15: 临时
 *   z16: 广播常量
 */
template<int R, int C>
inline __attribute__((always_inline)) void TLJ_FORCE(
    TileFixed<R,C> &fx_out, TileFixed<R,C> &fy_out, TileFixed<R,C> &fz_out,
    const TileFixed<R,C> &dx_in, const TileFixed<R,C> &dy_in, const TileFixed<R,C> &dz_in,
    const TileFixed<R,C> &rsq_in, const LJParamsT<R,C> &params,
    /* 临时 Tile (SVE后端不需要实际使用, 但保持接口一致) */
    TileFixed<R,C> &, TileFixed<R,C> &, TileFixed<R,C> &, TileFixed<R,C> &,
    TileFixed<R,C> &, TileFixed<R,C> &, TileFixed<R,C> &) {

    svbool_t pg = svptrue_b32();

    /* 加载输入到 SVE 寄存器 */
    svfloat32_t dx  = svld1_f32(pg, dx_in.data);
    svfloat32_t dy  = svld1_f32(pg, dy_in.data);
    svfloat32_t dz  = svld1_f32(pg, dz_in.data);
    svfloat32_t rsq = svld1_f32(pg, rsq_in.data);

    /* 条件逆: ir = 1/rsq (under predicate) */
    svbool_t valid = svand_b_z(pg,
        svcmplt_f32(pg, rsq, svdup_f32(params.cutoff_sq)),
        svcmpgt_f32(pg, rsq, svdup_f32(params.min_rsq)));
    svfloat32_t ir = svdiv_f32_z(valid, svdup_f32(1.0f), rsq);

    /* s2 = sigma_sq * ir */
    svfloat32_t s2 = svmul_f32_z(valid, svdup_f32(params.sigma_sq), ir);

    /* s6 = s2 * s2 * s2 */
    svfloat32_t s6 = svmul_f32_z(valid, svmul_f32_z(valid, s2, s2), s2);

    /* s12 = s6 * s6 */
    svfloat32_t s12 = svmul_f32_z(valid, s6, s6);

    /* fr = 24*eps * (2*s12 - s6) * ir */
    svfloat32_t fr = svmul_f32_z(valid,
        svsub_f32_z(valid, svmul_f32_z(valid, svdup_f32(2.0f), s12), s6),
        ir);
    fr = svmul_f32_z(valid, fr, svdup_f32(24.0f * params.epsilon));

    /* fx = fr * dx, fy = fr * dy, fz = fr * dz */
    svfloat32_t fx = svmul_f32_z(valid, fr, dx);
    svfloat32_t fy = svmul_f32_z(valid, fr, dy);
    svfloat32_t fz = svmul_f32_z(valid, fr, dz);

    /* 存储输出 (全部 valid 外的位置为 0, 由 _z 清零) */
    svst1_f32(pg, fx_out.data, fx);
    svst1_f32(pg, fy_out.data, fy);
    svst1_f32(pg, fz_out.data, fz);
}

/* ====================================================================
 * ================================================================== *
 *  Generic CPU Backend (编译器自定向量化)                              *
 *  用于不支持 SVE 的平台 (x86, 旧ARM 等)                             *
 * ================================================================== *
 * ==================================================================== */
#else /* !PTO_BACKEND_SVE */

/* --- TLOAD --- */
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

/* --- TFILL --- */
template<int R, int C>
inline __attribute__((always_inline)) void TFILL(TileFixed<R,C> &dst, float value) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) dst.data[i] = value;
}

/* --- TMUL --- */
template<int R, int C>
inline __attribute__((always_inline)) void TMUL(TileFixed<R,C> &dst,
    const TileFixed<R,C> &src0, const TileFixed<R,C> &src1) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) dst.data[i] = src0.data[i] * src1.data[i];
}

/* --- TADD --- */
template<int R, int C>
inline __attribute__((always_inline)) void TADD(TileFixed<R,C> &dst,
    const TileFixed<R,C> &src0, const TileFixed<R,C> &src1) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) dst.data[i] = src0.data[i] + src1.data[i];
}

/* --- TSUB --- */
template<int R, int C>
inline __attribute__((always_inline)) void TSUB(TileFixed<R,C> &dst,
    const TileFixed<R,C> &src0, const TileFixed<R,C> &src1) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) dst.data[i] = src0.data[i] - src1.data[i];
}

/* --- TDIV --- */
template<int R, int C>
inline __attribute__((always_inline)) void TDIV(TileFixed<R,C> &dst,
    const TileFixed<R,C> &src0, const TileFixed<R,C> &src1) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) {
        dst.data[i] = (src1.data[i] != 0.0f) ? src0.data[i] / src1.data[i] : 0.0f;
    }
}

/* --- TPBC --- */
template<int R, int C>
inline __attribute__((always_inline)) void TPBC(TileFixed<R,C> &dx, float box, float inv_box) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) dx.data[i] -= box * rintf(dx.data[i] * inv_box);
}

/* --- TCONDITIONAL_INV --- */
template<int R, int C>
inline __attribute__((always_inline)) void TCONDITIONAL_INV(TileFixed<R,C> &ir,
    const TileFixed<R,C> &rsq, float csq, float eps = 1e-8f) {
    PTO_CPU_VECTORIZE_LOOP
    for (int i = 0; i < C; i++) {
        ir.data[i] = (rsq.data[i] < csq && rsq.data[i] > eps) ? 1.0f / rsq.data[i] : 0.0f;
    }
}

/* --- TLJ_FORCE (Generic) --- */
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

#endif /* PTO_BACKEND_SVE */

} // namespace cpu
} // namespace pto

#endif /* PTO_GROMACS_CORE_HPP */
