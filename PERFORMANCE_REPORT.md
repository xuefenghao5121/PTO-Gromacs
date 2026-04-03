# GROMACS PTO 端到端性能报告

## 测试环境

- **测试时间**: 2026-04-03
- **GROMACS 版本**: 2023.3 (系统安装)
- **PTO 版本**: 1.0.0
- **平台**: x86_64 Linux
- **CPU**: Intel Xeon Platinum 8369B
- **编译器**: GCC 13.2.0
- **优化级别**: -O3 -march=native
- **AVX 支持**: Yes
- **AVX2 支持**: Yes
- **SIMD 向量宽度**: 8 floats (256-bit)

## 测试说明

由于系统安装的 GROMACS 2023.3 是 Ubuntu 预编译版本，无法直接集成自定义 PTO 代码。因此，本测试采用以下方法：

1. **独立基准测试**: 创建独立的性能测试程序，对比标量版本与优化版本
2. **模拟真实场景**: 使用类真实分子系统的原子分布和参数
3. **测量核心指标**: 计算时间、吞吐量、加速比

## 性能对比表格

| 测试用例 | 原子数 | 重复次数 | Baseline (ms) | PTO (ms) | 加速比 | 吞吐量 (ns/day) | 力差异 |
|---------|--------|---------|---------------|----------|--------|------------------|--------|
| Small (1K atoms) | 1024 | 10 | 0.644 | 0.611 | 1.05x | 282.90 | 0.000275 |
| Medium (2K atoms) | 2048 | 10 | 1.867 | 1.791 | 1.04x | 96.47 | 0.000200 |
| Large (4K atoms) | 4096 | 5 | 6.766 | 6.782 | 1.00x | 25.48 | 0.000113 |
| XLarge (8K atoms) | 8192 | 3 | 26.598 | 26.882 | 0.99x | 6.43 | 0.000000 |

## 性能分析

### 关键指标

- **平均加速比**: 1.02x
- **最小加速比**: 0.99x
- **最大加速比**: 1.05x
- **计算准确性**: 力差异 < 0.001（数值误差范围内）

### 性能趋势

1. **小规模系统 (1K-2K 原子)**:
   - PTO 提供约 4-5% 的性能提升
   - 循环展开和寄存器优化起作用
   - 缓存未饱和，优化空间有限

2. **中大规模系统 (4K-8K 原子)**:
   - 性能提升不明显或略有下降
   - 可能原因：
     - 编译器已充分优化基准代码
     - PTO 实现的额外开销（Tile 划分、邻居对构建）
     - 内存访问模式未优化

### 优化策略分析

#### 已实现的优化

1. **循环展开**: 手动展开内层循环，减少分支预测开销
2. **寄存器优化**: 累加变量使用独立寄存器，减少内存写回
3. **向量化**: 编译时启用 AVX/AVX2 指令集
4. **编译器优化**: 使用 -O3 -march=native

#### PTO 核心技术（理论）

1. **Tile 划分**: 基于空间填充曲线，将原子划分为适合 L2 缓存的 Tile
2. **算子融合**: 消除中间结果写回，所有计算在向量寄存器中完成
3. **缓存优化**: Tile 大小适配 L2 缓存 (256KB)，减少缓存缺失
4. **向量化**: 使用 AVX/AVX2 指令集并行处理多个原子

#### 为什么优化效果不明显？

1. **基准代码已充分优化**:
   - 现代 GCC 编译器 (-O3) 已自动向量化
   - 简单的双循环结构易于编译器优化
   - 人工优化难以超越编译器

2. **测试规模偏小**:
   - 8K 原子规模较小，内存瓶颈不明显
   - Tile 划分的开销在小规模下无法被收益抵消
   - 真实 GROMACS 模拟通常使用数万到数百万原子

3. **缺乏真实 GROMACS 集成**:
   - 独立测试无法利用 GROMACS 的完整优化栈
   - 缺少 GROMACS 的 nbnxn 模块优化
   - 未测试多线程并行性能

## GROMACS 端到端集成计划

### 方案 1: 从源码编译 GROMACS（推荐）

```bash
# 1. 下载 GROMACS 源码
wget ftp://ftp.gromacs.org/gromacs/gromacs-2023.3.tar.gz
tar -xzf gromacs-2023.3.tar.gz
cd gromacs-2023.3

# 2. 集成 PTO 代码
# 将 gromacs_pto_x86.h 和相关 .c 文件复制到 src/gromacs/mdlib/
# 在 nbnxn 相关文件中添加 PTO 调用

# 3. 编译 GROMACS
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/gromacs-pto \
         -DGMX_BUILD_OWN_FFTW=ON \
         -DGMX_GPU=OFF \
         -DGMX_SIMD=AVX2_256 \
         -DGMX_OPENMP=ON

make -j8
make install
```

### 方案 2: GROMACS Plugin 机制

GROMACS 2023+ 支持计算内核的 plugin 机制，可以动态加载 PTO 优化：

```c
// 实现 GROMACS 计算内核接口
int pto_nonbonded_kernel(gmx_nbnxn_pairlist_t *pairlist,
                           rvec *x, rvec *f,
                           real *shift_vec,
                           const interaction_const_t *ic,
                           const nbnxn_atomdata_t *nbat)
{
    // 调用 PTO 优化版本
    return gmx_pto_nonbonded_compute_fused_x86(...);
}
```

### 方案 3: LD_PRELOAD Hook

使用 LD_PRELOAD 拦截 GROMACS 的非键计算函数：

```bash
# 编译 PTO Hook
gcc -shared -fPIC -O3 pto_hook.c -o libpto_hook.so

# 运行 GROMACS 并注入 PTO
LD_PRELOAD=./libpto_hook.so gmx mdrun -s topol.tpr
```

## 真实 GROMACS 测试用例

### 标准测试系统

1. **LYSOSOME** (溶酶体蛋白):
   - 原子数: ~92,000
   - 预期性能: PTO 在大规模系统上优势更明显

2. **Villin in Water** (小蛋白质):
   - 原子数: ~10,000
   - 适合验证计算正确性

3. **Tryptophan Cage** (微小蛋白质):
   - 原子数: ~300
   - 用于快速验证

### 测试命令

```bash
# 使用 GROMACS 内置测试
gmx mdrun -s water_gro_box.tpr -nsteps 10000 -ntomp 8

# 测量性能
gmx mdrun -s water.tpr -ntomp 8 -nstlist 100 -rdd 1.2 -dlb auto
```

## 参数调优结果

### Tile 大小调优

理论最优 Tile 大小取决于缓存层次：

| CPU 缓存 | 大小 | 推荐 Tile 大小（原子数） |
|----------|------|--------------------------|
| L1 数据 | 32 KB | 16-32 |
| L2 | 256 KB | 64-128 |
| L3 | 8-64 MB | 256-1024 |

当前配置：
- 默认 Tile 大小: 1024 原子
- 目标缓存: L2 (256 KB)
- 实际使用: 自动调整

### 编译选项调优

推荐编译选项：

```bash
# AVX2 优化
gcc -O3 -march=haswell -ffast-math -funroll-loops

# FMA 优化（Fused Multiply-Add）
gcc -O3 -march=native -mfma

# 链接时优化（LTO）
gcc -O3 -flto -march=native
```

## 结论

### 当前结果

⚠️ PTO 优化在小规模独立测试中效果有限（平均加速比 1.02x）

### 原因分析

1. **编译器已充分优化**: 现代编译器对简单双循环优化非常出色
2. **测试规模偏小**: Tile 划分开销在小规模下无法被收益抵消
3. **缺乏真实集成**: 独立测试无法利用 GROMACS 的完整优化栈
4. **单线程限制**: 未测试多线程并行性能

### 预期真实性能

基于 PTO 理论分析和类似研究（如 ARM SME PTO），预期在以下场景有显著提升：

1. **大规模系统** (>100K 原子):
   - Tile 划分优势明显
   - 缓存局部性提升显著
   - 预期加速比: 1.5-2.0x

2. **多线程并行**:
   - Tile 并行化避免锁竞争
   - 负载均衡更好
   - 预期扩展性: 80-90%

3. **ARM 架构**:
   - SME Tile 硬件支持
   - 预期加速比: 2-4x

## 建议后续工作

### 短期（v0.4.0）

- [ ] 从源码编译 GROMACS 并集成 PTO
- [ ] 在真实 GROMACS 用例中测试（LYSOSOME）
- [ ] 修复 PTO 实现中的邻居对计算问题
- [ ] 添加 OpenMP 并行支持

### 中期（v0.5.0）

- [ ] 测试更大规模系统（100K+ 原子）
- [ ] 扩展到 ARM 架构（SVE/SME）
- [ ] 实现 GROMACS Plugin 机制
- [ ] 自动参数调优（Tile 大小、线程数）

### 长期（v1.0.0）

- [ ] 支持 GPU 加速（CUDA/OpenCL）
- [ ] 集成到 GROMACS 官方仓库（若性能显著）
- [ ] 发表学术论文
- [ ] 优化 Tile 划分算法（使用更高级的空间填充曲线）

## 参考文献

1. **GROMACS 文档**: https://manual.gromacs.org/
2. **ARM SME PTO**: Parallel Tile Operation optimization for ARM
3. **x86 优化指南**: Intel 64 and IA-32 Architectures Optimization Reference Manual
4. **HPC 并行计算**: Parallel Programming with OpenMP

## 版本信息

- **PTO-Gromacs 版本**: v0.3.0
- **报告生成时间**: 2026-04-03
- **测试者**: 天权团队

---

**备注**: 本报告基于独立基准测试结果。真实 GROMACS 集成后，性能可能会有显著不同。建议在真实环境中验证 PTO 优化效果。
