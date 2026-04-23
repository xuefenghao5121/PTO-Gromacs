# PTO-GROMACS - GROMACS 非键相互作用 ARM SVE 优化

## 核心成果

| 版本 | Small T16 | Medium T32 | Large T32 | 说明 |
|------|-----------|------------|-----------|------|
| v5 (手写SVE) | 11.35x | **19.78x** | **13.09x** | Medium/Large 最优 |
| v8 (PTO-ISA) | **12.74x** | 16.43x | 10.67x | Small 最优，可移植 |

**Small 规模 v8 超过 v5 12%！**

## 关键发现

1. **`svptest_any`** - Medium/Large 规模节省 50% 时间（跳过无效原子）
2. **标量循环写回** - GCC 自动向量化比显式 SVE 更高效
3. **最优方案** - Small 用 v8，Medium/Large 用 v5

## 目录结构

```
code/
├── pto_e2e_v5.c              # 手写 SVE（Medium/Large 最优）
├── pto_e2e_v8_megakernel.cpp # PTO-ISA（Small 最优）
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