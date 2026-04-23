# PTO-GROMACS 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `pto_gromacs_core.hpp` | PTO-ISA 核心头文件（多后端架构） |
| `pto_e2e_v5.c` | 手写 SVE 基准版本（最优 Medium/Large 性能） |
| `pto_e2e_v8_megakernel.cpp` | PTO-ISA 实现版本（最优 Small 性能） |
| `Makefile` | 编译规则 |

## 编译

```bash
# 编译 v5 (手写 SVE)
gcc -O3 -march=armv8.6-a+sve -msve-vector-bits=256 \
    -ffast-math -funroll-loops -fopenmp \
    -o pto_e2e_v5 pto_e2e_v5.c -lm

# 编译 v8 (PTO-ISA)
g++ -O3 -march=armv8.6-a+sve -msve-vector-bits=256 \
    -ffast-math -funroll-loops -fopenmp -std=c++17 \
    -o pto_e2e_v8_megakernel pto_e2e_v8_megakernel.cpp -lm
```

## 运行

```bash
# Small 规模测试
OMP_NUM_THREADS=16 ./pto_e2e_v5 box_small.gro 1.0 200
OMP_NUM_THREADS=16 ./pto_e2e_v8_megakernel box_small.gro 1.0 200

# Medium 规模测试
OMP_NUM_THREADS=32 ./pto_e2e_v5 box_medium.gro 1.0 200
OMP_NUM_THREADS=32 ./pto_e2e_v8_megakernel box_medium.gro 1.0 200
```

## 性能对比

| 版本 | Small T16 | Medium T32 | Large T32 |
|------|-----------|------------|-----------|
| v5 (手写SVE) | 11.35x | **19.78x** | **13.09x** |
| v8 (PTO-ISA) | **12.74x** | 16.43x | 10.67x |

**Small 规模 v8 超过 v5 12%！**

## 使用建议

- **Small 规模（<5000 原子）**：使用 v8
- **Medium/Large 规模（>5000 原子）**：使用 v5
