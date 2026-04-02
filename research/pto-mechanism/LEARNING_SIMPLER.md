# simpler 深度学习笔记

**学习时间**: 2026-04-02  
**仓库来源**: github.com/hw-native-sys/simpler  
**学习状态**: ✅ 完成

---

## 📋 第 1 步：理解定位

### 核心定位

**PTO Runtime** = 任务运行时执行框架

- **三程序模型**：Host `.so` + AICPU `.so` + AICore `.o`
- **任务依赖图**：构建和执行任务依赖图
- **协调执行**：AICPU 和 AICore 协调执行

### 在整个系统中的角色

```
PyPTO-Lib (原语库)
    ↓
PyPTO (Python DSL)
    ↓
PTOAS (编译器)
    ↓
pto-isa (指令集库)
    ↓
simpler (运行时框架) ← 当前学习
    ↓
硬件执行 (NPU)
```

### 主要职责

| 职责 | 描述 | 实现位置 |
|------|------|---------|
| **设备管理** | 管理设备操作 | `src/{arch}/platform/*/host/` |
| **任务调度** | 调度任务到 AICore | `src/{arch}/platform/*/aicpu/` |
| **内核执行** | 在 AICore 上执行内核 | `src/{arch}/platform/*/aicore/` |
| **内存管理** | 设备内存分配和传输 | `src/{arch}/platform/*/host/` |

---

## 📚 第 2 步：三程序模型

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Application                        │
│              (examples/scripts/run_example.py)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    Python Bindings   (ctypes)      Device I/O
    bindings.py
         │                │                │
         ▼                ▼                ▼
┌──────────────────┐  ┌──────────────────┐
│   Host Runtime   │  │   Binary Data    │
│ (src/{arch}/     │  │  (AICPU + AICore)│
│  platform/)      │  └──────────────────┘
├──────────────────┤         │
│ DeviceRunner     │         │
│ Runtime          │    Loaded at runtime
│ MemoryAllocator  │         │
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

---

### 程序 1: Host Runtime

**位置**：`src/{arch}/platform/*/host/`

**核心组件**：
- `DeviceRunner`：单例，管理设备操作
- `MemoryAllocator`：设备张量内存管理
- `pto_runtime_c_api.h`：纯 C API（供 Python 绑定）

**核心职责**：
- 分配/释放设备内存
- Host ↔ Device 数据传输
- AICPU 内核启动和配置
- AICore 内核注册和加载
- 运行时执行流程协调

**编译产物**：`.so` 共享库

---

### 程序 2: AICPU Kernel

**位置**：`src/{arch}/platform/*/aicpu/`

**核心组件**：
- `kernel.cpp`：内核入口点和握手协议
- 运行时特定执行器：`src/{arch}/runtime/*/aicpu/`

**核心职责**：
- 初始化与 AICore 核心的握手协议
- 识别初始就绪任务（fanin=0）
- 将就绪任务分派到空闲 AICore 核心
- 跟踪任务完成并更新依赖
- 持续执行直到所有任务完成

**编译产物**：`.so` 设备二进制

---

### 程序 3: AICore Kernel

**位置**：`src/{arch}/platform/*/aicore/`

**核心组件**：
- `kernel.cpp`：任务执行内核（add, mul 等）
- 运行时特定执行器：`src/{arch}/runtime/*/aicore/`

**核心职责**：
- 通过握手缓冲区等待任务分配
- 读取任务参数和内核地址
- 使用 PTO ISA 执行内核
- 发送任务完成信号
- 轮询下一个任务或退出信号

**编译产物**：`.o` 目标文件

---

## 🔬 第 3 步：API 层次

### Layer 1: C++ API

**位置**：`src/{arch}/platform/*/host/device_runner.h`

```cpp
DeviceRunner& runner = DeviceRunner::Get();
runner.Init(device_id, num_cores, aicpu_bin, aicore_bin, pto_isa_root);
runner.AllocateTensor(bytes);
runner.CopyToDevice(device_ptr, host_ptr, bytes);
runner.Run(runtime);
runner.Finalize();
```

---

### Layer 2: C API

**位置**：`src/{arch}/platform/include/host/pto_runtime_c_api.h`

```c
int DeviceRunner_Init(device_id, num_cores, aicpu_binary, aicpu_size,
                      aicore_binary, aicore_size, pto_isa_root);
int DeviceRunner_Run(runtime_handle, launch_aicpu_num);
int InitRuntime(runtime_handle);
int FinalizeRuntime(runtime_handle);
int DeviceRunner_Finalize();
```

---

### Layer 3: Python API

**位置**：`python/bindings.py`

```python
Runtime = bind_host_binary(host_binary)
runtime = Runtime()
runtime.initialize()
launch_runtime(runtime, aicpu_thread_num=1, block_dim=1,
               device_id=device_id, aicpu_binary=aicpu_bytes,
               aicore_binary=aicore_bytes)
runtime.finalize()
```

---

## 📊 第 4 步：执行流程

### Phase 1: Python Setup

```
Python run_example.py
  │
  ├─→ RuntimeCompiler.compile("host", ...) → host_binary (.so)
  ├─→ RuntimeCompiler.compile("aicpu", ...) → aicpu_binary (.so)
  ├─→ RuntimeCompiler.compile("aicore", ...) → aicore_binary (.o)
  │
  └─→ bind_host_binary(host_binary)
       └─→ RuntimeLibraryLoader(host_binary)
            └─→ CDLL(host_binary) ← 加载 .so 到内存
```

---

### Phase 2: Initialization

```
runner.init(device_id, num_cores, aicpu_binary, aicore_binary, pto_isa_root)
  │
  ├─→ DeviceRunner_Init (C API)
  │    ├─→ Initialize CANN device
  │    ├─→ Allocate device streams
  │    ├─→ Load AICPU binary to device
  │    ├─→ Register AICore kernel binary
  │    └─→ Create handshake buffers (one per core)
  │
  └─→ DeviceRunner singleton ready
```

---

### Phase 3: Runtime Building

```
runtime.initialize()
  │
  └─→ InitRuntime (C API)
       └─→ InitRuntimeImpl (C++)
            ├─→ Build task dependency graph
            ├─→ Allocate task arguments
            ├─→ Initialize task states
            └─→ Prepare handshake buffers
```

---

### Phase 4: Execution

```
runner.run(runtime)
  │
  └─→ DeviceRunner_Run (C API)
       └─→ Launch AICPU kernel
            │
            └─→ AICPU Task Scheduler
                 ├─→ Identify ready tasks (fanin=0)
                 ├─→ Dispatch to idle AICore cores
                 ├─→ AICore executes kernel
                 ├─→ Signal completion
                 ├─→ Update dependencies
                 └─→ Repeat until all tasks done
```

---

## 🎯 第 5 步：运行时变体

### 三种运行时

| 运行时 | 图构建位置 | 用途 |
|--------|-----------|------|
| `host_build_graph` | Host CPU | 开发、调试 |
| `aicpu_build_graph` | AICPU（设备） | 减少 Host-Device 传输 |
| `tensormap_and_ringbuffer` | AICPU（设备） | 生产工作负载 |

---

### 1. host_build_graph

**特点**：
- 任务依赖图在 Host CPU 上构建
- 适合开发和调试
- 易于观察和控制

**适用场景**：
- 原型开发
- 算法验证
- 性能分析

---

### 2. aicpu_build_graph

**特点**：
- 任务依赖图在 AICPU 上构建
- 减少 Host-Device 数据传输
- 更高效的执行

**适用场景**：
- 中等规模工作负载
- 需要减少 Host 开销

---

### 3. tensormap_and_ringbuffer

**特点**：
- 使用 TensorMap 和 RingBuffer 优化
- 最高性能
- 生产级实现

**适用场景**：
- 大规模生产工作负载
- 高吞吐量要求

---

## 🔧 第 6 步：握手协议

### 核心机制

**握手缓冲区**：每个 AICore 核心一个

```
┌─────────────────────────────────────┐
│        Handshake Buffer             │
├─────────────────────────────────────┤
│ task_id: int                        │
│ kernel_addr: void*                  │
│ args: void*                         │
│ status: enum (IDLE/RUNNING/DONE)    │
└─────────────────────────────────────┘
```

### 执行流程

```
AICPU Scheduler                    AICore Worker
     │                                  │
     ├─→ Write task to buffer           │
     │   (task_id, kernel_addr, args)   │
     │                                  │
     │                                  ├─→ Poll buffer
     │                                  ├─→ Read task
     │                                  ├─→ Execute kernel
     │                                  ├─→ Write DONE status
     │                                  │
     ├─→ Poll DONE status               │
     ├─→ Update dependencies            │
     └─→ Dispatch next task             │
```

---

## ✅ 学习总结

### 核心收获

1. **三程序模型**：Host `.so` + AICPU `.so` + AICore `.o`
2. **核心组件**：
   - Host Runtime：设备管理、内存分配
   - AICPU Kernel：任务调度器
   - AICore Kernel：计算内核
3. **API 层次**：C++ → C → Python
4. **执行流程**：Setup → Init → Build Graph → Execute
5. **运行时变体**：host_build_graph、aicpu_build_graph、tensormap_and_ringbuffer

### 与整个系统的关系

```
PyPTO-Lib (原语库)
    ↓ 使用
PyPTO (Python DSL)
    ↓ 编译
PTOAS (编译器)
    ↓ 生成
pto-isa (指令集库)
    ↓ 加载
simpler (运行时框架)
    ↓ 执行
硬件 (NPU)
```

### 完整学习路径总结

| 序号 | 仓库 | 核心概念 | 学习状态 |
|------|------|---------|---------|
| 1 | **PTOAS** | 编译器、自动同步插入 | ✅ 90% |
| 2 | **pto-isa** | 90+ 指令、Tile 类型系统 | ✅ 100% |
| 3 | **pypto** | Python DSL、Tile 级编程 | ✅ 100% |
| 4 | **pypto-lib** | 张量级原语、Incore Scope | ✅ 100% |
| 5 | **simpler** | 三程序模型、任务调度 | ✅ 100% |

---

**学习状态**: ✅ 完成  
**整体进度**: ✅ 100%（5/5 仓库学习完成）
