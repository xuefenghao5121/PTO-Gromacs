# GROMACS PTO ARM SVE/SME 优化 - 第一阶段开发成果

## 项目说明

本项目是基于ARM+SME/SVE纯CPU环境下，GROMACS非键相互作用PTO优化的第一阶段实现。

根据优化分析报告，第一阶段完成P0优先级的方案一：**非键相互作用优化开发**

## 已完成工作

### 1. Tile划分设计 ✓
- [x] 自适应Tile大小计算，基于L2缓存大小
- [x] 空间分块保持空间局部性
- [x] Tile邻居对构建（基于距离裁剪）
- [x] 支持可变Tile大小

### 2. 全流程算子融合 ✓
- [x] 融合坐标加载 -> 距离计算 -> LJ力 -> 静电力 -> 力累加整个流程
- [x] 消除中间结果内存写回
- [x] 一次Tile遍历完成所有计算
- [x] 保留OpenMP多核并行

### 3. SVE向量化实现 ✓
- [x] 利用ARM SVE可变长度向量
- [x] SVE距离平方计算
- [x] SVE LJ范德华力计算
- [x] SVE短程静电力计算
- [x] 自动适配不同SVE向量长度（128-2048位）

### 4. SME寄存器利用 ✓
- [x] SME可用性检测
- [x] 坐标加载到SME Tile寄存器
- [x] 力从SME Tile寄存器存储
- [x] 支持运行时动态启用/禁用SME

## 文件结构

```
code/
├── gromacs_pto_arm.h        # 头文件 - 公共接口定义
├── gromacs_pto_tiling.c      # Tile划分实现
├── gromacs_pto_sve.c         # SVE向量化计算实现
├── gromacs_pto_sme.c         # SME Tile寄存器利用实现
├── Makefile                  # Makefile
├── tests/
│   └── test_nonbonded.c      # 完整单元测试
└── README.md                 # 本文档
```

## 编译方法

### 环境要求

- ARMv9架构CPU
- GCC 12+ 或 Clang 15+
- Linux内核5.10+ （支持SVE/SME）
- 编译选项需要启用：`-march=armv9-a+sve+sme`

### 交叉编译（在x86主机上）

```bash
# 安装AArch64交叉编译器
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# 交叉编译
make CC=aarch64-linux-gnu-gcc clean all

# 复制编译结果到ARM目标机运行
scp tests/test_nonbonded user@arm-host:/tmp/
```

### 原生编译（在ARM机器上）

```bash
make clean all
make info
make test
```

## 接口说明

### 基础使用流程

```c
// 1. 初始化配置
gmx_pto_config_t config;
gmx_pto_config_init(&config);
config.tile_size_atoms = gmx_pto_auto_tile_size(total_atoms, 512);

// 2. 创建Tile划分
gmx_pto_nonbonded_context_t context;
gmx_pto_create_tiling(total_atoms, coords, &config, &context);

// 3. 构建邻居对
gmx_pto_build_neighbor_pairs(&context, coords, cutoff);

// 4. 设置参数
context.params.cutoff_sq = cutoff * cutoff;
context.params.charges = charges;
// ... 其他参数

// 5. 融合计算（整个流程一次调用）
gmx_pto_atom_data_t atom_data = {total_atoms, coords, forces};
gmx_pto_nonbonded_compute_fused(&context, &atom_data);

// 6. 清理
gmx_pto_destroy_tiling(&context);
```

## 设计要点

### Tile划分策略
- 每个Tile大小建议32-1024原子，默认64原子
- 根据L2缓存大小自动选择最佳大小
- 空间分块保持局部性，减少跨Tile邻居搜索

### 算子融合收益
- 原始流程：多次函数调用 + 多次内存读写
- PTO融合：一次加载坐标，所有计算完成后一次写回
- 消除中间结果内存访问，估计节省20-30%内存带宽

### SVE向量化设计
- 使用可变长度intrinsics，同一二进制适应不同向量长度
- 按SVE向量长度分组处理原子对
- 使用谓词处理边界，无需分支
- 自动利用更长向量获得更高性能

### SME利用
- Tile坐标直接存储在SME Tile寄存器中
- 减少对L1缓存的占用，提高缓存命中率
- 可用性检测，硬件不支持时自动降级到SVE

## 预期性能收益

基于分析报告，在典型ARMv9（512位SVE）芯片上：

- 非键组件提速：+25-45%
- 整体应用提速：+28-40%
- 能效提升：相同性能下功耗降低15-25%

详细分析请参考：[arm-sve-sve-analysis-report.md](../arm-sve-sve-analysis-report.md)

## 单元测试

单元测试覆盖：
- 配置初始化
- 自动Tile大小计算
- SVE向量长度查询
- SME可用性检测
- Tile划分创建
- 邻居对构建
- SVE距离计算对比参考实现
- SVE LJ力计算对比参考实现
- 完整融合计算流程验证

## 作者

天权团队 - 第一阶段开发
