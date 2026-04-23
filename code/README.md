# PTO-GROMACS 代码目录

## 文件

| 文件 | 说明 |
|------|------|
| `pto_e2e_v5.c` | 手写 SVE（Medium/Large 最优） |
| `pto_e2e_v8_megakernel.cpp` | Mega-kernel（Small 最优） |
| `pto_e2e_v11_ptoisa_chain.cpp` | PTO-ISA 算子链（可移植，性能较差） |
| `pto_gromacs_core.hpp` | PTO-ISA 核心接口 |

## 性能对比

| 版本 | Small T16 | Medium T32 | 说明 |
|------|-----------|------------|------|
| v5 | 11.35x | **19.78x** | Medium/Large 最优 |
| v8 | **12.74x** | 16.43x | Small 最优 |
| v11 | 27.44x | 44.83x | PTO-ISA 算子链 |

## 关键发现

**v11（真正的 PTO-ISA 算子链）比 v8 慢 91%！**

原因：
- v11 每个 PTO-ISA 算子都有 `svld1` + `svst1`，中间结果写回 Tile 内存
- v8 的 mega-kernel 全程在 SVE 寄存器中，零中间写回

**结论**：PTO-ISA 算子组合的可移植性优势，在性能上不如 mega-kernel。

## 编译

```bash
gcc -O3 -march=armv8.6-a+sve -msve-vector-bits=256 -ffast-math -fopenmp \
    -o pto_e2e_v5 pto_e2e_v5.c -lm

g++ -O3 -march=armv8.6-a+sve -msve-vector-bits=256 -ffast-math -fopenmp -std=c++17 \
    -o pto_e2e_v8 pto_e2e_v8_megakernel.cpp -lm
```

## 建议

- **Small 规模**：用 v8
- **Medium/Large 规模**：用 v5