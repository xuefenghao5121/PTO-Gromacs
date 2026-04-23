# PTO-GROMACS 代码目录

## 文件

| 文件 | 说明 |
|------|------|
| `pto_e2e_v5.c` | 手写 SVE（Medium/Large 最优） |
| `pto_e2e_v8_megakernel.cpp` | PTO-ISA（Small 最优） |
| `pto_gromacs_core.hpp` | PTO-ISA 核心接口 |

## 编译

```bash
gcc -O3 -march=armv8.6-a+sve -msve-vector-bits=256 -ffast-math -fopenmp \
    -o pto_e2e_v5 pto_e2e_v5.c -lm

g++ -O3 -march=armv8.6-a+sve -msve-vector-bits=256 -ffast-math -fopenmp -std=c++17 \
    -o pto_e2e_v8 pto_e2e_v8_megakernel.cpp -lm
```

## 运行

```bash
OMP_NUM_THREADS=16 ./pto_e2e_v5 box_small.gro 1.0 200
OMP_NUM_THREADS=32 ./pto_e2e_v5 box_medium.gro 1.0 200
```

## 性能

| 版本 | Small T16 | Medium T32 | Large T32 |
|------|-----------|------------|-----------|
| v5 | 11.35x | **19.78x** | **13.09x** |
| v8 | **12.74x** | 16.43x | 10.67x |

**建议**: Small 用 v8，Medium/Large 用 v5