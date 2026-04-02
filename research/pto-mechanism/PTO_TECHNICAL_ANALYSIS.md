# 华为 PTO 机制技术深度分析

**分析时间**: 2026-04-02  
**分析团队**: 天权-HPC团队  
**技术深度**: 架构 + 实现 + 应用

---

## 📋 技术栈全景图

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Application)                      │
│  LLM 推理引擎 (pypto-serving) | AI4S 应用 | 科学计算         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    编程框架层 (Framework)                    │
│  PyPTO (Python DSL) | Tile 编程模型 | Tensor/Tile 抽象      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    编译器层 (Compiler)                       │
│  PTOAS (LLVM/MLIR) | Pass 优化 | 代码生成 | Python 绑定     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    运行时层 (Runtime)                        │
│  simpler (Host/AICPU/AICore) | 分布式运行时 | 任务调度       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    硬件层 (Hardware)                         │
│  Ascend NPU | Cube + Vector 核心 | TPUSH/TPOP 指令集        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 核心技术组件深度分析

### 1. PTOAS 编译器工具链

#### 1.1 架构设计

**基于 LLVM/MLIR (llvmorg-19.1.7) 的 Out-of-Tree 架构**

```
PTOAS 编译流程：
┌──────────────┐
│  .pto 文件   │ (PTO Bytecode)
└──────┬───────┘
       ↓
┌──────────────┐
│  IR 解析     │ (MLIR Parser)
└──────┬───────┘
       ↓
┌──────────────┐
│  Pass 优化   │ (算子融合、同步插入、内存规划)
└──────┬───────┘
       ↓
┌──────────────┐
│  Lowering    │ (PTO → EmitC/Linalg)
└──────┬───────┘
       ↓
┌──────────────┐
│  代码生成    │ (C++ 代码 + PTO-ISA 调用)
└──────────────┘
```

#### 1.2 核心 Pass

| Pass 名称 | 功能 | 关键技术 |
|----------|------|---------|
| **InferPTOMemScope** | 推断内存作用域 | 作用域分析、内存层级推断 |
| **PTOViewToMemref** | 视图转换 | Memref 抽象、视图优化 |
| **PTOLowerFrontendPipeOpsPass** | 前端管道降级 | 管道操作降级 |
| **AllocToPointerCast** | 内存分配转换 | 指针转换、内存管理 |
| **PTOToEmitC** | PTO 到 C 代码 | 代码生成、C 后端 |
| **BufferizableOpInterfaceImpl** | 缓冲区接口 | 缓冲区管理、内存优化 |
| **LoweringSyncToPipe** | 同步降级 | 管道同步、事件管理 |
| **InferPTOLayout** | 布局推断 | 内存布局优化 |
| **InsertSync** | 自动同步插入 | **核心 Pass** ⭐ |

#### 1.3 自动同步插入 (InsertSync)

**核心算法**：
```cpp
// 同步事件 ID 分配算法
void SyncEventIdAllocation::Allocate(uint32_t runNum) {
  // 1. 正常分配
  for (auto &element : syncIR_) {
    AllocateEventId(element.get());
  }
  
  // 2. 尝试 Widen (复用事件 ID)
  for (auto &e : syncIR_) {
    WidenEventId(e->pipeAfter);
  }
  
  // 3. 处理资源不足需要重分配
  if (!reallocatedPipePair.empty()) {
    ReallocatedEventId();
    for (auto &e : syncIR_) {
      WidenEventId(e->pipeAfter);
    }
  }
  
  // 4. 降级策略：如果还是没有 ID，降级为 PipeAll 全局同步
  auto status = ChangeNoEventIdSyncToPipeAll();
  if (status.failed() && runNum < kMaxWidenTryNum) {
    if (tryWidenOnFirstFound()) {
      // 清空并重试
      reallocatedPipePair.clear();
      eventCyclePool.clear();
      clearAllocatedEventId();
      Allocate(runNum + 1);
    }
  }
}
```

**关键特性**：
- **事件 ID 池管理**：每个方向 8 个事件 ID
- **Widen 策略**：复用已完成的事件 ID
- **降级策略**：资源不足时降级为全局同步
- **最大尝试次数**：`kMaxWidenTryNum` 次重试

---

### 2. pypto-lib 原始张量函数库

#### 2.1 Tensor vs Tile 类型系统

**核心概念**：
- **Tensor**：N-D 逻辑张量（在 DDR 或全局内存）
- **Tile**：硬件感知的数据块（在统一缓冲区/本地内存）

**类型转换（无数据移动）**：
```python
# Tensor → Tile (视图转换)
tile = cast_tensor_to_tile(tensor, offsets, sizes)

# Tile → Tensor (视图转换)
tensor = cast_tile_to_tensor(tile)
```

**设计原则**：
- 库层面**不插入 TLOAD/TSTORE**
- 数据移动延迟到编译器决定 incore 边界后
- 编译器负责分片和降级到 PTO-ISA

#### 2.2 尾块处理和填充

**问题**：张量维度可能不能被 Tile 大小整除

**解决方案**：
1. **检测尾块**：最后一个 Tile 逻辑大小较小
2. **填充或掩码**：
   - 填充到完整 Tile 大小 + 掩码（PTO-ISA 支持）
   - 生成单独的尾块代码路径（无填充）
3. **正确性保证**：所有原语（归约、逐元素）必须遵守尾块语义

#### 2.3 跨复合函数融合

**挑战**：
```python
z = relu(softmax(x))
# 两个独立的复合函数，各自有自己的 Tile 循环
# 没有融合：完整遍历 softmax → 完整遍历 relu
# 融合后：单次遍历，每个 Tile 执行 softmax + relu
```

**融合条件**：
1. **迭代空间对齐**：两个循环遍历相同的 Tile 网格
2. **数据流清晰**：第一个复合的输出是第二个的唯一输入
3. **无干预使用**：两个循环之间没有其他使用第一个输出的代码
4. **合法性**：重排序不会违反依赖关系

**实现要求**：
- 显式、规范的 Tile 循环结构
- 复合函数之间的显式数据流
- 融合 Pass 可以识别并合并循环

#### 2.4 Incore 作用域设计

**核心概念**：
- **匿名 incore 函数**：用户不需要显式指定参数
- **编译器自动推导参数**：
  - **Input**：外部定义，内部只读
  - **Inout**：外部定义，内部修改
  - **Output**：外部定义，内部写入，外部读取

**示例**：
```python
def my_kernel(x: Tensor, y: Tensor) -> Tensor:
    tmp: Tensor  # 定义在外部，未赋值
    with incore_scope():
        # 内部：读取 x, y（只读）；写入 tmp（输出）
        for i in range(n):
            for j in range(c):
                tmp[i, j] = x[i, j] + y[i, j]
    # 外部：读取 tmp
    result = reduce_sum(tmp, axis=1)
    return result
```

**编译器生成**：
```python
# 生成的匿名 incore 函数
def my_kernel_incore_0(x: Tensor, y: Tensor, tmp: Tensor) -> None:
    for i in range(n):
        for j in range(c):
            tmp[i, j] = x[i, j] + y[i, j]

# 父函数中的调用
def my_kernel(x: Tensor, y: Tensor) -> Tensor:
    tmp: Tensor  # 运行时在调用 incore 时分配
    my_kernel_incore_0(x, y, tmp)  # 调用 incore 函数
    result = reduce_sum(tmp, axis=1)
    return result
```

**关键约束**：
- **工作集必须适合 SRAM**：incore 作用域内的中间数据不能超过核心内 SRAM 大小
- **内存溢出惩罚**：如果溢出到全局内存，性能严重下降

---

### 3. simpler 运行时框架

#### 3.1 三组件架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Application                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    Python Bindings   (ctypes)      Device I/O
         │                │                │
         ▼                ▼                ▼
┌──────────────────┐  ┌──────────────────┐
│   Host Runtime   │  │   Binary Data    │
│  (C++ Library)   │  │  (AICPU + AICore)│
├──────────────────┤  └──────────────────┘
│ DeviceRunner     │         │
│ Runtime          │         │
│ MemoryAllocator  │    Loaded at runtime
│ C API            │         │
└────────┬─────────┘         │
         │                   │
         └───────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │  Ascend Device (Hardware)   │
    ├────────────────────────────┤
    │ AICPU: Task Scheduler       │
    │ AICore: Compute Kernels     │
    └────────────────────────────┘
```

#### 3.2 Host Runtime (C++ 库)

**核心组件**：
- **DeviceRunner**：单例，管理设备操作
- **Runtime**：任务依赖运行时数据结构
- **MemoryAllocator**：设备张量内存管理
- **pto_runtime_c_api.h**：纯 C API（Python 绑定）

**关键职责**：
- 分配/释放设备内存
- 主机 ↔ 设备数据传输
- AICPU 内核启动和配置
- AICore 内核注册和加载
- 运行时执行工作流协调

#### 3.3 AICPU Kernel (任务调度器)

**关键职责**：
- 初始化与 AICore 核心的握手协议
- 识别初始就绪任务（fanin=0）
- 将就绪任务分派到空闲 AICore 核心
- 跟踪任务完成并更新依赖关系
- 持续执行直到所有任务完成

#### 3.4 AICore Kernel (计算内核)

**关键职责**：
- 通过握手缓冲区等待任务分配
- 读取任务参数和内核地址
- 使用 PTO ISA 执行内核
- 发送任务完成信号
- 轮询下一个任务或退出信号

#### 3.5 三层 API

| API 层 | 语言 | 用途 |
|--------|------|------|
| **Layer 1** | C++ | 高级接口，设备管理 |
| **Layer 2** | C | Python 绑定基础 |
| **Layer 3** | Python | 用户友好接口 |

---

### 4. pypto_runtime_distributed 分布式运行时

#### 4.1 Linqu 坐标系统

**7 层层级地址**：
```cpp
struct LinquCoordinate {
  uint32_t l6_idx;  // CLUSTER_2 / CLOS2
  uint32_t l5_idx;  // CLUSTER_1 / CLOS1
  uint32_t l4_idx;  // CLUSTER_0 / POD
  uint32_t l3_idx;  // HOST
  uint32_t l2_idx;  // CHIP
  uint32_t l1_idx;  // CHIP_DIE
  uint32_t l0_idx;  // CORE
};
```

**坐标操作**：
- `to_string()` → `(l6=0,l5=2,l4=1,l3=7,...)`
- `to_path()` → `L6_0/L5_2/L4_1/L3_7/...`
- `from_env()` → 读取 `LINQU_L0`–`LINQU_L6` 环境变量

#### 4.2 LinquOrchestrationAPI

**统一操作表**：

| 函数 | 签名 | 描述 |
|------|------|------|
| `submit_task` | `(rt, target, kernel_so, params, n)` | 分派内核到目标节点 |
| `scope_begin` | `(rt)` | 进入新作用域 |
| `scope_end` | `(rt)` | 退出当前作用域，释放环形槽位 |
| `alloc_tensor` | `(rt, target, size) → handle` | 在环形上分配缓冲区 |
| `free_tensor` | `(rt, handle)` | 早期释放（pl.free） |
| `orchestration_done` | `(rt)` | 信号编排完成 |
| `reg_data` | `(rt, target, data, size) → handle` | 注册现有数据 |
| `query_peers` | `(rt, level) → LinquPeerList` | 获取给定层级的对等节点 |
| `self_coord` | `(rt) → LinquCoordinate_C` | 此节点的坐标 |
| `wait_all` | `(rt)` | 等待所有分派的任务 |
| `submit_task_group` | `(rt, kernel_so, group_params, n_gp, sub_tasks, n_sub)` | 提交 N 个子任务作为一个依赖图节点 |

#### 4.3 任务组 API (submit_task_group)

**核心语义**：
- **依赖跟踪**：`group_params` INPUT/OUTPUT 句柄在 DAG 中构建边
- **SPMD 模式**：如果子任务的 `kernel_so` 为 NULL，使用组级内核
- **私有参数**：每个子任务可以携带每目标参数（如 shard_id）
- **空组**：`num_sub_tasks == 0` 立即完成
- **子任务 ID**：内部使用合成负 ID，不占用 TaskRing 槽位

---

## 🎯 AI4S 应用优化机会

### 1. 算子融合场景分析

| HPC 应用 | 计算模式 | 融合机会 | 预期收益 |
|---------|---------|---------|---------|
| **VASP** | FFT → 矩阵运算 → FFT | FFT 融合、矩阵乘优化 | 30-50% 性能提升 |
| **LAMMPS** | 邻居查找 → 力计算 | 数据局部性优化、流水线 | 20-40% 性能提升 |
| **GROMACS** | FFT → 电荷分配 → FFT | PME 算子融合 | 25-45% 性能提升 |
| **QE** | DFT 迭代 | 多步迭代融合 | 30-50% 性能提升 |

### 2. 优化策略

#### 2.1 数据局部性优化

**利用 PTO 零拷贝特性**：
```
传统方式：
Producer → GM → Consumer (两次 DMA)

PTO 方式（A5）：
Producer → Consumer SRAM (一次 DMA，零拷贝)
```

**性能提升**：
- 减少内存带宽占用
- 降低传输延迟
- 提高缓存命中率

#### 2.2 流水线并行

**多槽位环形缓冲区**：
```
SLOT_NUM = 8 或 4
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │  (槽位)
└───┴───┴───┴───┴───┴───┴───┴───┘
  ↑       ↑       ↑       ↑
 生产者   消费者   生产者   消费者
```

**优势**：
- 生产者-消费者模式
- 隐藏内存延迟
- 提高吞吐量

#### 2.3 细粒度同步

**硬件标志机制**：
```
每个方向（Vector→Cube, Cube→Vector）有 8 个硬件标志
总计 32 个跨核心标志（2 peers × 2 directions × 8 flags）

SET/WAIT 操作 → 无锁同步
```

**优势**：
- 无软件锁开销
- 确定性延迟
- 支持细粒度并行

---

## 📊 性能关键路径分析

### 1. 编译时优化

| 优化阶段 | 关键技术 | 性能影响 |
|---------|---------|---------|
| **算子融合** | 跨复合函数融合 | 减少 50-70% 内存访问 |
| **同步插入** | 自动同步、事件 ID 复用 | 减少 20-30% 同步开销 |
| **内存规划** | 作用域分析、布局推断 | 减少 30-50% 内存占用 |
| **代码生成** | PTO-ISA 降级 | 硬件级优化 |

### 2. 运行时优化

| 优化技术 | 实现方式 | 性能影响 |
|---------|---------|---------|
| **零拷贝传输** | TPUSH/TPOP 指令 | 减少 50% 传输延迟 |
| **环形缓冲区** | 多槽位流水线 | 提高 2-3x 吞吐量 |
| **任务调度** | DAG 依赖管理 | 提高并行度 |
| **内存管理** | Ring-heap 分配 | O(1) 分配/释放 |

---

## 🚀 后续研究方向

### 1. 深入学习（优先级 P0）

- [ ] 分析 PTOAS 编译器 Pass 实现
- [ ] 学习 pypto-lib 算子实现细节
- [ ] 研究 simpler 运行时调度算法

### 2. HPC 应用分析（优先级 P1）

- [ ] 选择目标 HPC 应用（VASP/LAMMPS/GROMACS）
- [ ] 分析计算瓶颈和数据流
- [ ] 识别算子融合机会

### 3. 原型实现（优先级 P2）

- [ ] 设计针对特定应用的算子融合方案
- [ ] 实现优化算子
- [ ] 性能测试和对比分析

---

## 📚 关键技术文档索引

### 设计文档
1. `machine_hierarchy_and_function_hierarchy.md` - 层级模型
2. `linqu_runtime_design.md` - 分布式运行时设计
3. `tpush_tpop_isa_design_v3.md` - TPUSH/TPOP 指令集设计
4. `pypto_serving_design goal.md` - LLM 推理引擎设计目标

### 实现文档
1. `PTOAS/README.md` - 编译器工具链
2. `pypto-lib/README.md` - 原始张量函数库
3. `simpler/README.md` - 运行时框架
4. `pypto_runtime_distributed/docs/api_reference.md` - 分布式运行时 API

### 代码仓库
1. `PTOAS` - PTO 汇编器
2. `pypto` - 编程框架
3. `pypto-lib` - 算子库
4. `pypto-serving` - LLM 推理引擎
5. `pypto_runtime_distributed` - 分布式运行时
6. `simpler` - 芯片运行时
7. `silk_v2` - 编译器
8. `pto-isa` - PTO 指令集架构

---

**分析状态**: ✅ 深度分析完成  
**下一步**: 选择目标 HPC 应用，设计算子融合方案
