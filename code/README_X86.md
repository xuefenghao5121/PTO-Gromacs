# GROMACS PTO x86 实现说明

## 项目概述

本项目实现了 GROMACS 非键相互作用优化的 x86 版本，使用 PTO（Parallel Tile Operation）技术进行性能优化。

## 实现特性

### 1. 核心功能

- **Tile 划分** - 自适应的原子分块，适配 x86 缓存层次（L1/L2/L3）
- **算子融合** - 消除中间内存读写，提升计算效率
- **AVX/AVX2 向量化** - 256-bit SIMD 指令集支持
- **OpenMP 并行化** - 多线程计算
- **PyPTO 集成接口** - 提供与 PyPTO 库的集成点

### 2. 文件结构

```
code/
├── gromacs_pto_x86.h      # x86 PTO 头文件（API 定义）

├── gromacs_pto_x86.c      # Tile 划分 + 融合计算核心
│                           # - 空间填充曲线分块
│                           # - 自适应 Tile 大小计算
│                           # - 邻居对构建
│                           # - 全流程融合计算
│
├── gromacs_pto_avx.c       # AVX/AVX2 向量化内核
│                           # - AVX 距离计算
│                           # - AVX LJ 力计算
│                           # - AVX 库仑力计算
│                           # - FMA 优化（AVX2）
│
└── gromacs_pto_pypto.c    # PyPTO 集成文档和接口
                            # - PyPTO 使用说明
                            # - 占位函数（C 环境）
```

### 3. 测试文件

```
code/tests/
└── test_nonbonded_x86.c   # x86 单元测试
                             # - CPU 检测测试
                             # - 配置初始化测试
                             # - Tile 创建测试
                             # - 邻居对构建测试
                             # - AVX 计算正确性测试
                             # - 完整流程测试
```

## 编译和测试

### 编译

```bash
cd code/
make x86
```

### 运行测试

```bash
cd code/
./tests/test_nonbonded
```

预期输出：

```
========================================
GROMACS PTO x86 单元测试
========================================

测试1: CPU检测
  AVX支持: 是
  AVX2支持: 是
  向量宽度: 8 floats
  通过

[...]

========================================
测试结果: 8 通过, 0 失败
========================================
```

### 清理

```bash
make clean
```

## PyPTO 使用说明

### PyPTO 版本

```
pypto 0.1.2
```

### Python 示例代码

```python
import pypto
import numpy as np

# 创建 PyPTO 上下文
ctx = pypto.Context()

# 准备数据
n_atoms = 1000
coords = np.random.rand(n_atoms, 3).astype(np.float32)
forces = np.zeros((n_atoms, 3), dtype=np.float32)

# 创建 PyPTO Tensor
coords_tensor = pypto.tensor(coords, dtype=pypto.DT_FP32)
forces_tensor = pypto.tensor(forces, dtype=pypto.DT_FP32)

# 定义计算内核
def compute_nonbonded(coords_i, coords_j, cutoff_sq):
    dx = coords_i[0] - coords_j[0]
    dy = coords_i[1] - coords_j[1]
    dz = coords_i[2] - coords_j[2]
    rsq = dx*dx + dy*dy + dz*dz
    
    # LJ 力计算
    sigma = 0.3
    epsilon = 0.5
    sigma_over_r_sq = sigma * sigma / rsq
    t6 = sigma_over_r_sq ** 3
    t12 = t6 * t6
    f_over_r = 24.0 * epsilon * (2.0 * t12 - t6) / rsq
    
    # 力分量
    fx = f_over_r * dx
    fy = f_over_r * dy
    fz = f_over_r * dz
    
    return fx, fy, fz

# 编译和优化（PyPTO 自动处理）
# - Tile 划分
# - 算子融合
# - 向量化
# - 并行化
kernel = pypto.compile(compute_nonbonded)

# 执行计算
result = kernel(coords_tensor, coords_tensor, cutoff_sq=1.5)
```

### C + PyPTO 混合使用

```c
#include "gromacs_pto_x86.c"

// 初始化配置
gmx_pto_config_x86_t config;
gmx_pto_config_x86_init(&config);

// 检查 PyPTO 是否可用
if (gmx_pto_pypto_is_available()) {
    char *version = gmx_pto_pypto_get_version();
    printf("PyPTO version: %s\n", version);
    free(version);
    
    // 使用 PyPTO 优化的 Tile 划分
    ret = gmx_pto_pypto_create_tiling(n_atoms, coords, &config, &context);
    
    // 使用 PyPTO 融合计算
    ret = gmx_pto_pypto_fused_compute(&context, &atom_data, true);
} else {
    // 使用原生 C 实现
    ret = gmx_pto_create_tiling_x86(n_atoms, coords, &config, &context);
    ret = gmx_pto_nonbonded_compute_fused_x86(&context, &atom_data);
}
```

## 性能特点

### 算子融合优势

**传统实现（无融合）**：

```
坐标加载 → 写入临时数组
    ↓
距离计算 → 写入临时数组
    ↓
力计算 → 写入临时数组
    ↓
力累加 → 写回最终数组
```

**PTO 融合实现**：

```
坐标加载
    ↓
[距离 + 力计算 + 力累加] 全部在寄存器完成
    ↓
一次性写回最终数组
```

### 预期性能提升

- **内存访问减少**: ~50-70%（消除中间写回）
- **缓存利用率**: 提升（显著Tile 局部性）
- **向量化效率**: AVX2 8 floats 并行
- **多线程扩展**: OpenMP 线性扩展（小Tile）

## 架构设计

### Tile 划分

```
原子空间
    │
    ├─┬─┬─┬─ Tile 0 (64 atoms)
    │ │ │ │
    ├─┼─┼─┼─ Tile 1 (64 atoms)
    │ │ │ │
    └─┴─┴─┴─ Tile N (≤64 atoms)
```

每个 Tile:
- 适配 L2 缓存大小（~256 KB）
- 独立可并行计算
- 空间局部性显著

### 邻居对

```
Tile 0 ─── Tile 1
   │          │
   │      (neighbor pair)
   │          │
   └────── Tile 2
```

基于空间距离构建邻居对，避免全局遍历。

## 未来改进

1. **Hilbert 空间填充曲线** - 更好的空间局部性
2. **动态 Tile 大小调整** - 运行时自适应
3. **AVX-512 支持** - 更宽的向量（16 floats）
4. **负载均衡优化** - 密度感知 Tile 划分
5. **PyPTO 深度集成** - 通过 CFFI 连接

## 技术栈

- **C11** - 核心实现
- **AVX/AVX2 Intrinsics** - 向量化
- **OpenMP** - 并行化
- **Python 3** - PyPTO 集成
- **GCC/Clang** - 编译器

## 参考实现

- ARM SVE/SME 版本：`gromacs_pto_arm.h`, `gromacs_pto_sve.c`, `gromacs_pto_sme.c`
- ARM Tile 划分：`gromacs_pto_tiling.c`

本 x86 版本移植自 ARM 实现，核心原理一致：
- ARM SVE → x86 AVX/AVX2
- ARM SME Tile 寄存器 → x86 向量寄存器
- 缓存层次适配（L1/L2/L3）

## 作者

天权团队
GitHub: https://github.com/xuefenghao5121/PTO-Gromacs

## 许可证

与 GROMACS 项目一致（LGPL）
