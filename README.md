# PTO-GROMACS - GROMACS 基于华为 PTO 技术的 ARM SVE/SME 优化

## 项目概述

本项目是天权-HPC团队针对 GROMACS 分子动力学模拟应用，基于华为 PTO (Processing-in-Memory Tiling Optimization) 技术，在 ARMv9 架构上进行性能优化的研究项目。

**目标**: 通过 PTO 算子融合和 Tile 划分技术，结合 ARM SVE 可变长度向量化和 SME 矩阵扩展，提升 GROMACS 在 ARM 平台上的性能。

## 团队信息

- **团队**: 天权-HPC团队 (team_tianquan_hpc)
- **角色分工**:
  - 天权 (architect): 整体架构把控
  - 天璇 (hpc_expert): 文档整理与实现
  - 天玑 (ai4s_researcher): 内容验证与性能分析

## 目录结构

```
PTO-Gromacs/
├── README.md                     # 本文件 - 项目说明
├── SOUL.md                       # 团队定位和工作方式
├── TEAM.md                       # 团队成员和协作规则
├── DEV_LOG_PHASE1.md             # 第一阶段开发日志
├── WORK_LOG.md                   # 工作日志
├── TODO.md                       # 待办事项
├── full-analysis-report.md        # 完整分析报告
├── arm-sve-sve-analysis-report.md # ARM SVE/SME 优化分析报告
├── research/                     # 调研资料
│   ├── pto-mechanism/            # 华为 PTO 机制研究（仅文档，不包含上游源码）
│   ├── gromacs-hotspot-analysis.md  # GROMACS 热点分析
│   └── operator-fusion/          # 算子融合技术研究
├── designs/                      # 优化方案设计
│   ├── architecture/            # 架构设计
│   ├── algorithms/              # 算法设计
│   └── implementation/          # 实现设计
│   └── gromacs-pto-optimization.md # GROMACS PTO优化方案
├── code/                         # 第一阶段实现代码
│   ├── *.h                      # 头文件
│   ├── *.c                      # 实现代码
│   ├── Makefile                 # 编译规则
│   ├── README.md                # 代码说明
│   └── tests/                   # 单元测试和基准测试
├── benchmarks/                   # 性能测试
│   ├── results/                 # 测试结果
│   ├── scripts/                 # 测试脚本
│   └── reports/                 # 性能报告
└── docs/                         # 文档输出
    ├── technical-reports/       # 技术报告
    ├── best-practices/          # 最佳实践
    └── presentations/           # 演示文稿
```

## 当前状态 - 第一阶段完成 ✅

**阶段**: 第一阶段 - 分析报告与核心实现
**状态**: 已完成，等待推送到 GitHub

### 第一阶段已完成内容

1. ✅ PTO 机制原理学习与分析
2. ✅ GROMACS 热点分析 - 识别非键相互作用占 65-85% 计算时间
3. ✅ ARM SVE/SME 架构特性分析
4. ✅ 优化方案设计 - Tile 划分 + 全流程算子融合
5. ✅ 第一阶段代码实现 - 非键相互作用核心 kernel
   - Tile 划分 (自适应大小)
   - SVE 向量化计算
   - SME 寄存器利用
   - 单元测试覆盖
   - 性能基准测试
6. ✅ 完整分析报告文档

### 预期性能收益

| SVE向量长度 | 预期整体性能提升 | 非键组件提升 |
|-------------|-----------------|-------------|
| 128位 | +8-15% | +15-25% |
| 256位 | +18-28% | +25-35% |
| 512位 | +28-40% | +35-45% |
| 1024位 | +35-48% | +35-55% |

## 下一阶段

- 第二阶段: 集成到 GROMACS 主代码库
- 第三阶段: 完整性能测试与调优
- 第四阶段: 扩展到其他热点组件（PME、LINCS）

## 编译运行（第一阶段测试）

```bash
cd code
make clean all
make test        # 运行单元测试
make benchmark    # 运行性能基准测试
```

详见 [code/README.md](./code/README.md)

## 许可证

本项目为学术研究用途

---

**最后更新**: 2026-04-02
**第一阶段完成**: 2026-04-02
