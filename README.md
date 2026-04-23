# PTO-GROMACS - GROMACS 非键相互作用 ARM SVE 优化

## 核心成果

| 版本 | Small T16 | Medium T32 | Large T32 | 说明 |
|------|-----------|------------|-----------|------|
| v5 (手写SVE) | 11.35x | **19.78x** | **13.09x** | Medium/Large 最优 |
| v8 (mega-kernel) | **12.74x** | 16.43x | 10.67x | Small 最优 |
| v11 (PTO-ISA chain) | 27.44x | 44.83x | - | 真正的 PTO-ISA 算子组合 |

**Small 规模 v8 超过 v5 12%！**

## 关键发现

### 1. Mega-kernel vs PTO-ISA 算子链

| 方案 | Small T16 | 性能差距 | 原因 |
|------|-----------|----------|------|
| v8 (mega-kernel) | **12.70x** | 最优 | 全程在 SVE 寄存器，零中间写回 |
| v11 (PTO-ISA chain) | 27.44x | 慢 91% | 每个算子都有 load/store 开销 |

**结论**：PTO-ISA 算子组合的性能不如 mega-kernel，因为中间结果需要写回 Tile 内存。

### 2. `svptest_any` 优化效果

| 版本 | Medium T32 | 说明 |
|------|------------|------|
| v5 (svptest_any) | **19.53x** | 最优 |
| v5_noptest (无 svptest_any) | 37.28x | 慢 91% |

**结论**：`svptest_any` 在 Medium/Large 规模是有效优化。

### 3. 最优方案

- **Small 规模**：v8（mega-kernel，零中间写回）
- **Medium/Large 规模**：v5（svptest_any + 标量循环写回）

## 目录结构

```
code/
├── pto_e2e_v5.c              # 手写 SVE（Medium/Large 最优）
├── pto_e2e_v8_megakernel.cpp # Mega-kernel（Small 最优）
├── pto_e2e_v11_ptoisa_chain.cpp # PTO-ISA 算子链（可移植，性能较差）
└── pto_gromacs_core.hpp      # PTO-ISA 核心接口
```

## 编译运行

```bash
# 编译
gcc -O3 -march=armv8.6-a+sve -msve-vector-bits=256 -ffast-math -fopenmp \
    -o pto_e2e_v5 pto_e2e_v5.c -lm

g++ -O3 -march=armv8.6-a+sve -msve-vector-bits=256 -ffast-math -fopenmp -std=c++17 \
    -o pto_e2e_v8 pto_e2e_v8_megakernel.cpp -lm

# 运行
OMP_NUM_THREADS=16 ./pto_e2e_v5 box_small.gro 1.0 200
OMP_NUM_THREADS=16 ./pto_e2e_v8 box_small.gro 1.0 200
```

## 团队

天权-HPC团队 | 学术研究用途

---

**2026-04-23** | Small 规模 v8 超过 v5 12%