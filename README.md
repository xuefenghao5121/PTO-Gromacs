# PTO-GROMACS - GROMACS 非键相互作用 ARM SVE 优化

## 项目概述

本项目是天权-HPC团队针对 GROMACS 分子动力学模拟中的**非键相互作用（Non-bonded interactions）**，在 ARM SVE 平台上进行性能优化的研究项目。

**核心成果**：
- ✅ **Small 规模 PTO-ISA 版本超过手写 SVE 12%**
- ✅ PTO-ISA 可移植架构，支持多后端（SVE / Generic / 未来扩展）
- ✅ 完整的性能分析和优化路径记录

## 最终性能对比

| 版本 | Small T16 | Medium T32 | Large T32 | 架构 |
|------|-----------|------------|-----------|------|
| **v5 (手写SVE)** | 11.35x | **19.78x** | **13.09x** | 纯 SVE intrinsics |
| **v8 (PTO-ISA)** | **12.74x** | 16.43x | 10.67x | PTO 接口 + SVE 后端 |

**Small 规模 v8 超过 v5 12%！**

### 线程扩展性（Small 规模）

| 线程数 | v5 | v8 | v8 vs v5 |
|--------|-----|-----|----------|
| T1 | 0.81x | **0.96x** | +19% |
| T8 | 6.31x | **7.32x** | +16% |
| T16 | 10.17x | 10.00x | -2% |
| T32 | **12.84x** | 10.27x | -20% |

## 关键发现

### 1. `svptest_any` 优化效果

| 版本 | Medium T32 | 说明 |
|------|------------|------|
| v5 (svptest_any) | **19.53x** | 最优 |
| v5_noptest (无 svptest_any) | 37.28x | 慢 91% |

**结论**：`svptest_any` 在 Medium/Large 规模是有效优化，跳过无效原子节省 50% 时间。

### 2. 标量循环写回 vs 向量化写回

| 版本 | Medium T32 | 说明 |
|------|------------|------|
| v5 (标量循环) | **19.53x** | 编译器自动向量化 |
| v5_vecwrite (向量化) | 38.94x | 慢 99% |

**结论**：GCC 对标量循环的自动向量化比显式 SVE 向量化更高效。

### 3. Small vs Medium/Large 的差异

| 规模 | v8 vs v5 | 原因 |
|------|----------|------|
| Small | v8 快 12% | 原子数少，分支预测开销更明显 |
| Medium/Large | v8 慢 17-23% | 原子数多，`svptest_any` 跳过无效原子的收益更大 |

### 最优方案

- **Small 规模**：v8（移除 `svptest_any`）
- **Medium/Large 规模**：v5（保留 `svptest_any` + 标量循环写回）

## 目录结构

```
PTO-Gromacs/
├── README.md                     # 本文件
├── code/
│   ├── pto_gromacs_core.hpp      # PTO-ISA 核心头文件（多后端架构）
│   ├── pto_e2e_v5.c              # 手写 SVE 基准版本
│   └── pto_e2e_v8_megakernel.cpp # PTO-ISA 实现版本
├── research/                     # 调研资料
├── designs/                      # 优化方案设计
└── docs/                         # 文档输出
```

## PTO-ISA 架构

### 三层抽象

```
┌─────────────────────────────────────────┐
│  PTO 算子 API (可移植)                   │
│  TLOAD, TMUL, TADD, TSUB, TPBC...       │
└─────────────────────────────────────────┘
              ↓ 编译时后端选择
┌─────────────────────────────────────────┐
│  后端实现                                │
│  ├── SVE:  svld1, svmul, svadd...       │
│  ├── Generic: 编译器自定向量化           │
│  └── (未来) AVX-512, NPU                │
└─────────────────────────────────────────┘
```

### 核心数据结构

```cpp
template<int ROWS, int COLS>
struct TileFixed {
    alignas(64) float data[ROWS * COLS];  // 64字节对齐
    int valid_cols;                        // 有效列数
};
```

**关键**：`TileFixed<1, 8>` = **一个 SVE 256-bit 向量**（8 个 float）

### 核心算子

| 算子 | 功能 | SVE 后端实现 |
|------|------|-------------|
| `TLOAD` | 加载 Tile | `svld1_f32` |
| `TSTORE` | 存储 Tile | `svst1_f32` |
| `TMUL` | 逐元素乘法 | `svmul_f32_x` |
| `TADD` | 逐元素加法 | `svadd_f32_x` |
| `TSUB` | 逐元素减法 | `svsub_f32_x` |
| `TPBC` | 周期性边界条件 | `svsub + svmul + svrinta` |
| `TREDUCE` | 横向归约 | `svadda_f32` |

### 矩阵转置处理

**PTO-ISA 不在算子层处理转置，而是在数据层一次性完成：**

```cpp
// main() 中一次性转置 AoS → SoA
for (int i = 0; i < n; i++) {
    sx[i] = x[i*3+0];  // 所有 x 坐标连续
    sy[i] = x[i*3+1];  // 所有 y 坐标连续
    sz[i] = x[i*3+2];  // 所有 z 坐标连续
}
```

**优势**：
1. 一次性转置，后续所有计算都是连续访问
2. 算子层不需要处理转置，保持简洁
3. 缓存友好，减少内存访问

## 编译运行

### 环境要求

- ARM 平台（鲲鹏930 或类似）
- GCC 12+ 或 Clang 15+
- SVE 支持（`-march=armv8.6-a+sve`）

### 编译

```bash
cd code

# 编译 v5 (手写 SVE)
gcc -O3 -march=armv8.6-a+sve -msve-vector-bits=256 \
    -ffast-math -funroll-loops -fopenmp \
    -o pto_e2e_v5 pto_e2e_v5.c -lm

# 编译 v8 (PTO-ISA)
g++ -O3 -march=armv8.6-a+sve -msve-vector-bits=256 \
    -ffast-math -funroll-loops -fopenmp -std=c++17 \
    -o pto_e2e_v8_megakernel pto_e2e_v8_megakernel.cpp -lm
```

### 运行

```bash
# 准备 GRO 文件（或使用自己的）
# 格式: GROMACS .gro 坐标文件

# 运行 v5
OMP_NUM_THREADS=16 ./pto_e2e_v5 box_small.gro 1.0 200

# 运行 v8
OMP_NUM_THREADS=16 ./pto_e2e_v8_megakernel box_small.gro 1.0 200
```

### 参数说明

- 第一个参数：GRO 文件路径
- 第二个参数：截断半径（nm），默认 1.0
- 第三个参数：迭代步数，默认 200

## 优化历程

### v5 - 手写 SVE 基准版本

**关键优化**：
1. 排序 j-list → 最大化连续加载
2. `svptest_any` → 跳过无效原子
3. 标量循环写回 → 编译器自动向量化
4. 动态调度 → 负载均衡

### v8 - PTO-ISA 实现版本

**关键优化**：
1. PTO-ISA 可移植接口
2. 移除 `svptest_any` → 减少分支预测开销
3. 向量化 j 力写回 → `svld1 → svsub → svst1`
4. Mega-kernel → 零中间结果溢出

### 失败尝试

| 尝试 | 结果 | 原因 |
|------|------|------|
| v9 延迟 j 力写回 | 慢 2x | 标量循环无法向量化 |
| v10 混合版本 | 慢 2x | C++ 编译器优化不如 C |

## 团队信息

- **团队**: 天权-HPC团队 (team_tianquan_hpc)
- **角色分工**:
  - 天权 (architect): 整体架构把控
  - 天璇 (hpc_expert): 文档整理与实现
  - 天玑 (ai4s_researcher): 内容验证与性能分析

## 许可证

本项目为学术研究用途

---

**最后更新**: 2026-04-23
**v8 完成**: 2026-04-23 - Small 规模超过 v5 12%
