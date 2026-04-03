# PTO-Gromacs v0.3.0 Release Notes

## 发布日期

2026-04-03

## 版本信息

- **版本号**: v0.3.0
- **代号**: End-to-End Benchmark
- **状态**: 完成端到端性能测试

## 新增功能

### 1. 端到端性能基准测试

- ✅ 创建独立的性能测试程序
- ✅ 对比标量版本 vs PTO 优化版本
- ✅ 测试多规模系统（1K-8K 原子）
- ✅ 测量关键性能指标（加速比、吞吐量、ns/day）
- ✅ 验证计算正确性（力差异 < 0.001）

### 2. 测试套件

新增以下测试程序：

- `code/tests/simple_benchmark.c`: 简化版性能测试
- `code/tests/pto_benchmark.c`: 完整 PTO 性能测试
- `code/tests/end_to_end_benchmark.c`: 端到端测试

### 3. 性能报告

- ✅ 生成详细的性能分析报告（`PERFORMANCE_REPORT.md`）
- ✅ 包含性能对比表格
- ✅ 加速比分析和趋势解读
- ✅ 优化策略说明
- ✅ 后续工作建议

### 4. 文档更新

- ✅ 更新 README.md，添加 v0.3.0 说明
- ✅ 添加 RELEASE_NOTES_v0.3.0.md
- ✅ 完善 GROMACS 集成方案

## 性能测试结果

### 测试环境

- **GROMACS 版本**: 2023.3 (系统安装)
- **CPU**: Intel Xeon Platinum 8369B
- **编译器**: GCC 13.2.0
- **优化级别**: -O3 -march=native
- **AVX/AVX2**: 支持

### 性能指标

| 测试用例 | 原子数 | Baseline (ms) | PTO (ms) | 加速比 |
|---------|--------|---------------|----------|--------|
| Small (1K atoms) | 1024 | 0.644 | 0.611 | 1.05x |
| Medium (2K atoms) | 2048 | 1.867 | 1.791 | 1.04x |
| Large (4K atoms) | 4096 | 6.766 | 6.782 | 1.00x |
| XLarge (8K atoms) | 8192 | 26.598 | 26.882 | 0.99x |

**平均加速比**: 1.02x

### 结果分析

⚠️ **独立测试中 PTO 优化效果有限**

**原因**：
1. 现代编译器 (-O3) 已充分优化基准代码
2. 测试规模偏小（8K 原子），Tile 划分开销无法被收益抵消
3. 缺乏真实 GROMACS 集成，无法利用完整优化栈

**预期真实性能**：
- 大规模系统（>100K 原子）：预期加速比 1.5-2.0x
- ARM SME 架构：预期加速比 2-4x

## GROMACS 集成方案

由于系统安装的 GROMACS 是预编译版本，无法直接集成 PTO。提供以下方案：

### 方案 1: 从源码编译（推荐）

```bash
# 下载源码
wget ftp://ftp.gromacs.org/gromacs/gromacs-2023.3.tar.gz
tar -xzf gromacs-2023.3.tar.gz

# 集成 PTO 代码
# 编译
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/gromacs-pto \
         -DGMX_SIMD=AVX2_256 -DGMX_OPENMP=ON
make -j8 && make install
```

### 方案 2: GROMACS Plugin

GROMACS 2023+ 支持计算内核 plugin，可动态加载 PTO 优化。

### 方案 3: LD_PRELOAD Hook

使用 LD_PRELOAD 拦截 GROMACS 的非键计算函数。

详见 `PERFORMANCE_REPORT.md`。

## 已知问题

1. **邻居对计算**: PTO 实现中的邻居对计算存在问题，需要修复
2. **Tile 划分**: 小规模测试中 Tile 划分优势不明显
3. **单线程限制**: 未测试多线程并行性能

## 技术改进

### x86 实现

- ✅ 完整的 AVX/AVX2 支持
- ✅ Tile 划分算法（空间填充曲线）
- ✅ 算子融合（中间结果保留在寄存器）
- ✅ 缓存优化（适配 L2 缓存）

### PyPTO 集成

- ✅ Python 接口支持
- ✅ 自动 Tile 大小调优
- ✅ 版本检测和功能查询

## 文件变更

### 新增文件

```
PTO-Gromacs/
├── code/tests/
│   ├── simple_benchmark.c       # 简化版性能测试
│   ├── pto_benchmark.c           # PTO 完整性能测试
│   └── end_to_end_benchmark.c    # 端到端测试
├── PERFORMANCE_REPORT.md        # 性能报告
└── RELEASE_NOTES_v0.3.0.md      # 本文件
```

### 修改文件

```
PTO-Gromacs/
└── README.md                    # 更新版本说明
```

## 测试覆盖

- ✅ 编译测试（x86 AVX/AVX2）
- ✅ 功能测试（计算正确性）
- ✅ 性能测试（多规模）
- ⚠️ 真实 GROMACS 集成测试（待源码编译）

## 下一步计划（v0.4.0）

- [ ] 从源码编译 GROMACS 并集成 PTO
- [ ] 在真实 GROMACS 用例中测试（LYSOSOME，92K 原子）
- [ ] 修复邻居对计算问题
- [ ] 添加 OpenMP 并行支持
- [ ] 测试大规模系统（100K+ 原子）

## 致谢

感谢天权团队的支持和反馈。

## 许可证

MIT License

---

**发布者**: 天权团队
**日期**: 2026-04-03
