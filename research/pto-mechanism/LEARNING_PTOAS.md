# PTOAS 深度学习笔记

**学习时间**: 2026-04-02  
**仓库来源**: github.com/hw-native-sys/PTOAS  
**学习状态**: 🔄 进行中

---

## 📋 第 1 步：理解定位

### 核心定位

**PTOAS** 是 PTO（Programming Tiling Operator）的编译器工具链，负责将高层 PTO IR 编译为底层可执行代码。

### 在整个系统中的角色

```
上层 AI 框架 (PyPTO, TileLang, CuTile)
              ↓
         PTOAS 编译器 ← 当前学习
              ↓
      pto-isa 指令集库
              ↓
         硬件执行
```

### 主要职责

| 职责 | 描述 | 实现位置 |
|------|------|---------|
| **IR 解析与验证** | 解析 `.pto` 文件，验证语义正确性 | `lib/PTO/IR/` |
| **编译优化 Pass** | 算子融合、自动同步插入、内存规划 | `lib/PTO/Transforms/` |
| **代码生成** | PTO IR → EmitC/Linalg → C++ 代码 | `lib/PTO/Transforms/PTOToEmitC.cpp` |
| **Python 绑定** | 提供 Python 接口 | `lib/Bindings/Python/` |

---

## 📚 第 2 步：阅读 README

### 核心信息

- **基础框架**: LLVM/MLIR (llvmorg-19.1.7)
- **架构模式**: Out-of-Tree（不修改 LLVM 源码）
- **目标硬件**: 达芬奇架构（Da Vinci Architecture）
- **输入格式**: `.pto` 文件（PTO Bytecode）
- **输出格式**: C++ 代码（调用 pto-isa 库）

### 构建依赖

- **LLVM 版本**: llvmorg-19.1.7（严格依赖）
- **编译器**: GCC >= 9 或 Clang（支持 C++17）
- **构建系统**: CMake >= 3.20, Ninja
- **Python**: 3.8+（pybind11, numpy）

### 命令行工具

| 工具 | 功能 |
|------|------|
| `ptoas` | 主编译器（解析、优化、代码生成） |
| `ptobc` | 字节码工具 |

---

## 📁 第 3 步：分析目录结构

### 顶层目录

```
PTOAS/
├── include/PTO/        # 头文件和 TableGen 定义（20 个文件）
├── lib/PTO/            # 核心实现（30 个 .cpp 文件）
├── python/             # Python 模块
├── test/samples/       # 测试用例
├── tools/              # 命令行工具
└── docs/               # 文档
```

### 核心目录详解

#### include/PTO/

```
include/PTO/
├── IR/                 # PTO Dialect 定义
│   ├── PTODialect.h    # Dialect 主头文件
│   ├── PTOOps.td       # 操作定义（TableGen）
│   ├── PTOTypeDefs.td  # 类型定义（TableGen）
│   └── PTOAttrs.td     # 属性定义（TableGen）
└── Transforms/         # Pass 头文件
    ├── Passes.h        # Pass 声明
    └── InsertSync/     # 自动同步插入相关
```

#### lib/PTO/

```
lib/PTO/
├── IR/                 # Dialect 实现
│   ├── PTO.cpp         # 主实现（348KB，核心）
│   ├── PTOAttrs.cpp    # 属性实现
│   └── PTOTypeDefs.cpp # 类型实现
└── Transforms/         # Pass 实现
    ├── PTOToEmitC.cpp  # 代码生成（361KB，最大）
    ├── PTOViewToMemref.cpp # 视图转换（102KB）
    ├── PTOPlanMemory.cpp # 内存规划（74KB）
    ├── InferPTOMemScope.cpp # 内存作用域推断
    ├── InferPTOLayout.cpp # 布局推断
    └── InsertSync/     # 自动同步插入（核心创新）
```

---

## 🔬 第 4 步：深入核心代码

### 核心 1: PTO Dialect 定义

**文件**: `include/PTO/IR/PTODialect.h`

```cpp
// PTO Dialect 主头文件
#include "mlir/IR/Dialect.h"

#include "PTO/IR/PTODialect.h.inc"  // TableGen 生成
#include "PTO/IR/PTOOps.h.inc"      // TableGen 生成
```

**关键理解**：
- PTO Dialect 基于 MLIR Dialect 机制
- 使用 TableGen 定义操作和类型
- 自动生成 C++ 代码

---

### 核心 2: PTO 操作定义

**文件**: `include/PTO/IR/PTOOps.td`（4660 行）

**操作分类**（70+ 操作）：

#### 内存管理操作

| 操作 | 描述 | 示例 |
|------|------|------|
| `AllocTileOp` | 分配 Tile 缓冲区 | `pto.alloc_tile : !pto.tile_buf<...>` |
| `TFreeOp` | 释放 Tile | `pto.tfree %tile : !pto.tile_buf<...>` |
| `TPushToAicOp` | 推送到 AIC | `pto.tpush_to_aic %tile` |
| `TPopFromAicOp` | 从 AIC 拉取 | `pto.tpop_from_aic %tile` |

#### 计算操作

| 操作 | 描述 | 示例 |
|------|------|------|
| `TMatmulOp` | 矩阵乘 | `pto.tmatmul %a, %b -> %c` |
| `TAddOp` | 加法 | `pto.tadd %a, %b -> %c` |
| `TExpOp` | 指数 | `pto.texp %a -> %b` |

#### 同步操作

| 操作 | 描述 | 示例 |
|------|------|------|
| `SyncSetOp` | 设置同步事件 | `pto.sync_set %event_id` |
| `SyncWaitOp` | 等待同步事件 | `pto.sync_wait %event_id` |
| `BarrierOp` | 屏障 | `pto.barrier` |

---

### 核心 3: PTO 类型系统

**文件**: `include/PTO/IR/PTOTypeDefs.td`（223 行）

#### 核心类型

| 类型 | 语法 | 描述 |
|------|------|------|
| `!pto.ptr<elementType>` | `!pto.ptr<f16>` | 全局内存指针 |
| `!pto.tensor_view<d0 x d1 x elementType>` | `!pto.tensor_view<1024x512xf16>` | 张量视图 |
| `!pto.partition_tensor_view<d0 x d1 x elementType>` | `!pto.partition_tensor_view<16x16xf16>` | 分区张量视图 |
| `!pto.tile_buf<...>` | `!pto.tile_buf<loc=vec, dtype=f16, ...>` | Tile 缓冲区 |

#### tile_buf 参数详解

| 参数 | 类型 | 描述 | 示例 |
|------|------|------|------|
| `loc` | `vec/mat/left/right/acc/bias` | 内存域 | `loc=vec` |
| `dtype` | `f16/f32/i32/...` | 元素类型 | `dtype=f16` |
| `rows/cols` | `int64` | 物理行列数 | `rows=16, cols=16` |
| `v_row/v_col` | `int64` 或 `?` | 有效行列数 | `v_row=16, v_col=16` |
| `blayout` | `row_major/col_major` | 基础布局 | `blayout=row_major` |
| `slayout` | `none_box/row_major/col_major` | 次级布局 | `slayout=none_box` |
| `fractal` | `int32` | 分形大小 | `fractal=512` |
| `pad` | `0/...` | 填充值 | `pad=0` |

---

### 核心 4: 编译 Pass

#### Pass 1: PTOToEmitC（代码生成）

**文件**: `lib/PTO/Transforms/PTOToEmitC.cpp`  
**大小**: 361KB，9194 行  
**功能**: 将 PTO IR 降级到 EmitC Dialect，生成 C++ 代码

**关键理解**：
- 这是最复杂的 Pass
- 负责将高层 PTO 操作转换为底层 C++ 代码
- 需要处理内存布局、同步、流水线等

---

#### Pass 2: PTOPlanMemory（内存规划）

**文件**: `lib/PTO/Transforms/PTOPlanMemory.cpp`  
**大小**: 74KB，2036 行  
**功能**: 规划内存分配和生命周期

**关键理解**：
- NP-hard 问题
- 需要考虑内存复用
- 需要处理作用域

---

#### Pass 3: InsertSync（自动同步插入）⭐ 核心创新

**目录**: `lib/PTO/Transforms/InsertSync/`  
**功能**: 自动插入同步操作，管理事件 ID

**子模块**：

| 文件 | 行数 | 功能 |
|------|------|------|
| `PTOIRTranslator.cpp` | 722 | IR 转换 |
| `SyncEventIdAllocation.cpp` | 715 | 事件 ID 分配 ⭐ |
| `InsertSyncAnalysis.cpp` | 570 | 同步分析 |
| `SyncCodegen.cpp` | 401 | 同步代码生成 |
| `SyncCommon.cpp` | 306 | 公共工具 |

**核心算法**（SyncEventIdAllocation）：

```cpp
void SyncEventIdAllocation::Allocate(uint32_t runNum) {
  // 1. 正常分配事件 ID
  for (auto &element : syncIR_) {
    AllocateEventId(element.get());
  }
  
  // 2. Widen 策略（复用已完成的事件 ID）
  for (auto &e : syncIR_) {
    WidenEventId(e->pipeAfter);
  }
  
  // 3. 资源不足时重分配
  if (!reallocatedPipePair.empty()) {
    ReallocatedEventId();
    for (auto &e : syncIR_) {
      WidenEventId(e->pipeAfter);
    }
  }
  
  // 4. 降级策略：PipeAll 全局同步
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

**关键理解**：
- 每个方向有 8 个事件 ID
- Widen 策略：复用已完成的事件 ID
- 降级策略：资源不足时使用全局同步
- 最大尝试次数：`kMaxWidenTryNum`

---

## 📝 第 5 步：总结提炼

### 核心收获

1. **PTO Dialect 设计**：
   - 基于 MLIR Dialect 机制
   - 使用 TableGen 定义操作和类型
   - 70+ 操作，覆盖内存管理、计算、同步等

2. **编译流程**：
   ```
   .pto 文件 → IR 解析 → Pass 优化 → 代码生成 → C++ 代码
   ```

3. **核心 Pass**：
   - PTOToEmitC：代码生成（最复杂）
   - PTOPlanMemory：内存规划（NP-hard）
   - InsertSync：自动同步插入（核心创新）

4. **关键创新**：
   - 自动同步插入算法
   - 事件 ID 分配和复用
   - 降级策略（全局同步）

### 待深入理解

- [ ] PTOToEmitC 如何处理不同操作？
- [ ] 内存规划算法的具体实现？
- [ ] 同步插入的完整流程？
- [ ] 如何运行测试用例？

---

## 🚀 下一步

1. **阅读测试用例** - 理解实际使用方式
2. **分析 PTOToEmitC.cpp** - 理解代码生成逻辑
3. **深入 InsertSync** - 理解自动同步插入算法
4. **进入下一个仓库（pto-isa）** - 学习指令集架构

---

**学习状态**: 🔄 进行中（60% 完成）  
**下一步**: 阅读测试用例，理解实际使用方式
