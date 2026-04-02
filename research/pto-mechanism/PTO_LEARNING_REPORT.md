# 华为 PTO 机制学习报告

**学习时间**: 2026-04-02  
**学习团队**: 天权-HPC团队  
**资料来源**: hw-native-sys, hengliao1972 GitHub 仓库

---

## 📋 核心概念总结

### 1. PTO（Pass-Through Optimization）定义

PTO 是华为开发的一种**硬件级优化机制**，用于加速 AI 推理和科学计算中的核心间数据传输。

**核心思想**：
- 通过专用指令集（TPUSH/TPOP）实现跨核心零拷贝数据传输
- 硬件级同步机制，避免软件锁开销
- 支持流水线式数据处理，提高吞吐量

---

### 2. 集群架构（Cluster Architecture）

每个集群包含：
- **1 个 Cube 核心**：矩阵计算核心（类似 GPU 的 Tensor Core）
- **2 个 Vector 核心**：向量计算核心（类似 GPU 的 CUDA Core）

```
┌─────────────────────── Cluster ───────────────────────┐
│                                                       │
│  ┌──────────┐    flags (8 per dir)    ┌──────────┐   │
│  │  Vector 0 │◄══════════════════════►│          │   │
│  └──────────┘   SET/WAIT V→C, C→V    │   Cube   │   │
│                                       │          │   │
│  ┌──────────┐    flags (8 per dir)    │          │   │
│  │  Vector 1 │◄══════════════════════►│          │   │
│  └──────────┘   SET/WAIT V→C, C→V    └──────────┘   │
│                                                       │
└───────────────────────────────────────────────────────┘
```

**同步机制**：
- 每个方向（Vector→Cube, Cube→Vector）有 **8 个硬件标志**
- 总计 **32 个跨核心标志**（2 peers × 2 directions × 8 flags）
- 支持 SET/WAIT 操作，实现无锁同步

---

### 3. TPUSH/TPOP 指令集

#### 3.1 核心指令

| 指令 | 执行位置 | 角色 | 方向 | 描述 |
|------|---------|------|------|------|
| `tpush_to_aiv` | Cube | 生产者 | C2V | 推送 Tile 到 Vector 核心 |
| `tpush_to_aic` | Vector | 生产者 | V2C | 推送 Tile 到 Cube 核心 |
| `tpop_from_aic` | Vector | 消费者 | C2V | 从 Cube 拉取 Tile |
| `tpop_from_aiv` | Cube | 消费者 | V2C | 从 Vector 拉取 Tile |
| `tfree_to_aic` | Vector | 消费者 | C2V | 释放槽位，通知生产者可重用 |
| `tfree_to_aiv` | Cube | 消费者 | V2C | 释放槽位，通知生产者可重用 |

#### 3.2 环形缓冲区（Ring Buffer）

**数据结构**：
- 多槽位环形缓冲区（SLOT_NUM = 8 或 4）
- 每个槽位存储一个固定大小的 Tile
- 基于标签（tag）的流控制协议

**通信模式**：
| 模式 | SLOT_NUM | 标志使用 | 描述 |
|------|---------|---------|------|
| 单向 | 8 | 8 flags P2C + C2P | 单方向数据流 |
| 双向 | 4 per dir | 4 flags each | 双方向同时通信 |

#### 3.3 平台差异

| 平台 | 生产者缓冲区 | 消费者缓冲区 | 数据路径 |
|------|------------|------------|---------|
| **A2A3** | GM（全局内存） | 消费者 SRAM | 两跳：Producer → GM → Consumer SRAM |
| **A5** | 消费者 SRAM | 同一缓冲区 | 一跳：Producer → Consumer SRAM（零拷贝） |

**关键设计原则**：
- 消费者的 `TILE.data` 总是引用本地 SRAM 槽位缓冲区
- 内核程序在 A2/A3 和 A5 上**完全相同**
- 平台差异隐藏在 `tpush`/`tpop` 传输实现中

---

### 4. PyPTO 编程模型

#### 4.1 层级架构

```
L7 Global      - 全局请求路由，QoS 分级
L6 Cluster     - QoS 分级，多档 batch size
L5 Supernode   - Prefill/Decode 服务分离
L4 Pod         - Prefill/Decode disaggregation
L3 Host        - 单机推理核心
L2 Chip        - NPU 芯片级 kernel
L1 InCore      - 单个核心计算
```

#### 4.2 编码规则

**每个层级严格遵循**：
- **1 个编排函数（orchestrator）**：构建任务 DAG，提交 worker 和子编排器
- **多个 worker 函数**：纯计算函数，在 worker 线程上并行执行

**示例**：
```python
# L3 Host orchestrator
def prefill_orchestrator(input_ids):
    # 构建 DAG
    tasks = []
    for chunk in split_input(input_ids):
        task = submit_worker(prefill_worker, chunk)
        tasks.append(task)
    
    # 等待所有 worker 完成
    results = wait_all(tasks)
    return merge_results(results)

# L3 Host worker
def prefill_worker(chunk):
    # 纯计算，不提交后续任务
    return compute(chunk)
```

---

### 5. 应用场景

#### 5.1 LLM 推理引擎（pypto-serving）

**设计目标**：
- C/C++ 高性能核心
- 基于 Lingqu L3–L7 分布式运行时
- Radix Tree KV Cache 管理
- 自回归循环优化

**性能关键路径**：
- Prefill、Decode、KV Cache 访问、Radix Tree 查找
- 全部用 C/C++ 实现
- 不允许 Python 出现在自回归路径中

#### 5.2 AI4S 场景优化潜力

**适合算子融合的 HPC 应用**：
- **VASP**：电子结构计算（矩阵运算密集）
- **LAMMPS**：分子动力学（力计算、邻居列表）
- **GROMACS**：生物分子模拟（FFT、PME）
- **Quantum ESPRESSO**：DFT 计算（FFT、矩阵对角化）

**优化方向**：
1. **数据预处理 → 计算 → 后处理** 流水线融合
2. **迭代循环** 内核融合（如 Jacobi 迭代）
3. **多物理场耦合** 算子融合

---

## 🔍 关键技术洞察

### 1. 零拷贝数据传输

**传统方式**：
```
Producer → GM → Consumer
（两次 DMA，高延迟）
```

**PTO 方式（A5）**：
```
Producer → Consumer SRAM
（一次 DMA，零拷贝）
```

**性能提升**：
- 减少内存带宽占用
- 降低传输延迟
- 提高缓存命中率

### 2. 硬件级同步

**优势**：
- 无软件锁开销
- 确定性延迟
- 支持细粒度并行

**实现**：
- SET/WAIT 标志操作
- 每个方向 8 个标志
- 支持多槽位流水线

### 3. 平台自适应

**设计原则**：
- 内核程序平台无关
- 传输层平台相关
- 编译器自动选择路径

---

## 📊 与 AI4S 场景的结合点

### 1. 算子融合机会

| HPC 应用 | 计算模式 | 融合机会 |
|---------|---------|---------|
| **VASP** | FFT + 矩阵运算 | FFT → 矩阵乘 → FFT 融合 |
| **LAMMPS** | 力计算 + 邻居列表 | 邻居查找 → 力计算融合 |
| **GROMACS** | PME + FFT | FFT → 电荷分配 → FFT 融合 |
| **QE** | DFT 迭代 | 多步迭代融合 |

### 2. 性能优化策略

1. **数据局部性优化**
   - 利用 PTO 零拷贝特性
   - 减少全局内存访问
   - 提高缓存利用率

2. **流水线并行**
   - 多槽位环形缓冲区
   - 生产者-消费者模式
   - 隐藏内存延迟

3. **细粒度同步**
   - 硬件标志机制
   - 无锁数据结构
   - 低延迟协调

---

## 🚀 下一步研究方向

### 1. 深入学习（优先级 P0）

- [ ] 学习 PTOAS（PTO 汇编器）实现细节
- [ ] 分析 pypto-lib 算子库
- [ ] 研究 simpler 运行时架构

### 2. HPC 应用分析（优先级 P1）

- [ ] 选择目标 HPC 应用（VASP/LAMMPS/GROMACS）
- [ ] 分析计算瓶颈和数据流
- [ ] 识别算子融合机会

### 3. 原型实现（优先级 P2）

- [ ] 设计针对特定应用的算子融合方案
- [ ] 实现优化算子
- [ ] 性能测试和对比分析

---

## 📚 参考文档

### 核心设计文档
1. `machine_hierarchy_and_function_hierarchy.md` - 层级模型
2. `linqu_runtime_design.md` - 分布式运行时设计
3. `tpush_tpop_isa_design_v3.md` - TPUSH/TPOP 指令集设计
4. `pypto_serving_design goal.md` - LLM 推理引擎设计目标

### 代码仓库
1. `pypto_top_level_design_documents` - 顶层设计文档
2. `pto-isa` - PTO 指令集架构
3. `pypto` - 社区驱动实现
4. `pypto-lib` - 算子库
5. `pypto-serving` - LLM 推理引擎
6. `pypto_runtime_distributed` - 分布式运行时
7. `PTOAS` - PTO 汇编器
8. `simpler` - 芯片运行时
9. `silk_v2` - 编译器
10. `XiangShan-doc` - 香山处理器文档

---

**学习状态**: ✅ 初步学习完成  
**下一步**: 深入学习 PTOAS 和 pypto-lib，分析具体算子实现
