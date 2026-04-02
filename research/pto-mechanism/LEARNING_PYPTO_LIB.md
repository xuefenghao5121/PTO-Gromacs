# pypto-lib 深度学习笔记

**学习时间**: 2026-04-02  
**仓库来源**: github.com/hw-native-sys/pypto-lib  
**学习状态**: ✅ 完成

---

## 📋 第 1 步：理解定位

### 核心定位

**PyPTO-Lib** = 张量级原语库（类似 PyTorch ATen）

- **Tensor 级原语**：不是 Tile 级（Tile 级就是 PTO-ISA）
- **编译器负责 Tiling**：将 Tensor 操作分解为 Tile 循环
- **硬件无关**：不指定 incore vs orchestration

### 在整个系统中的角色

```
PyPTO-Lib (原语库) ← 当前学习
    ↓ 使用 pypto IR
PyPTO (Python DSL)
    ↓ 编译
PTOAS (编译器)
    ↓ 生成
pto-isa (指令集库)
    ↓ 执行
硬件 (NPU)
```

### 主要职责

| 职责 | 描述 | 实现位置 |
|------|------|---------|
| **原语定义** | 定义张量级原语集 | `README.md` |
| **示例实现** | 提供高性能算子示例 | `examples/` |
| **编译集成** | 通过 pypto IR 编译到 PTO-ISA | `pypto/` |

---

## 📚 第 2 步：核心概念

### 1. Tensor vs Tile

**关键理解**：Tensor 和 Tile 是**两个不同的类型**，需要通过 cast 操作转换。

#### cast_tensor_to_tile

```python
# Tensor → Tile（视图转换，无数据移动）
tile = cast_tensor_to_tile(tensor, offsets, sizes)
```

- **语义**：Tile 是 Tensor 的一个子区域视图
- **无数据复制**：只是逻辑描述符（shape, stride, base pointer）
- **延迟加载**：编译器决定何时插入 TLOAD

#### cast_tile_to_tensor

```python
# Tile → Tensor（逆视图转换）
tensor = cast_tile_to_tensor(tile)
```

- **语义**：Tile 对应的 Tensor 视图
- **类型兼容**：用于链式操作

---

### 2. Incore Scope 机制

**核心创新**：用 `with incore_scope():` 标记一个区域成为匿名 incore 函数。

#### 参数推导规则

| 参数类型 | 条件 | 内存分配 |
|---------|------|---------|
| **input** | 外部定义，scope 内只读 | 外部已分配 |
| **inout** | 外部定义并初始化，scope 内读写 | 外部分配 |
| **output** | 外部定义但未初始化，scope 内写入 | **运行时分配** |

#### 示例 1: 简单 Incore Scope

```python
def my_kernel(x: Tensor, y: Tensor) -> Tensor:
    tmp: Tensor   # 未初始化 → output
    with incore_scope():
        # x, y 只读 → input
        # tmp 写入 → output
        tmp = x + y
    result = reduce_sum(tmp, axis=1)  # scope 后读取 tmp
    return result
```

**编译器生成**：

```python
def my_kernel_incore_0(x: Tensor, y: Tensor, tmp: Tensor) -> None:
    tmp = x + y

def my_kernel(x: Tensor, y: Tensor) -> Tensor:
    tmp: Tensor   # 运行时分配
    my_kernel_incore_0(x, y, tmp)
    result = reduce_sum(tmp, axis=1)
    return result
```

#### 示例 2: 多输入 + Inout + Output

```python
def fused_update(x: Tensor, y: Tensor, scale: float) -> Tensor:
    acc: Tensor = zeros((n, c))   # 初始化 → inout
    out: Tensor                   # 未初始化 → output
    with incore_scope():
        # x, y, scale 只读 → input
        # acc 读写 → inout
        # out 写入 → output
        for i in range(n):
            for j in range(c):
                acc[i, j] += x[i, j] * scale
                out[i, j] = y[i, j] + acc[i, j]
    result = reduce_sum(out, axis=1) + reduce_sum(acc, axis=1)
    return result
```

**编译器生成**：

```python
def fused_update_incore_0(x: Tensor, y: Tensor, scale: float, 
                          acc: Tensor, out: Tensor) -> None:
    for i in range(n):
        for j in range(c):
            acc[i, j] += x[i, j] * scale
            out[i, j] = y[i, j] + acc[i, j]

def fused_update(x: Tensor, y: Tensor, scale: float) -> Tensor:
    acc = zeros((n, c))   # inout: 父函数创建并初始化
    out: Tensor           # output: 运行时分配
    fused_update_incore_0(x, y, scale, acc, out)
    result = reduce_sum(out, axis=1) + reduce_sum(acc, axis=1)
    return result
```

---

### 3. 原语集（类似 ATen）

#### 逐元素运算

| 原语 | 功能 | 数学定义 |
|------|------|---------|
| `add(x, y)` | 加法 | `z = x + y` |
| `sub(x, y)` | 减法 | `z = x - y` |
| `mul(x, y)` | 乘法 | `z = x * y` |
| `div(x, y)` | 除法 | `z = x / y` |
| `sqrt(x)` | 平方根 | `z = √x` |
| `exp(x)` | 指数 | `z = e^x` |
| `log(x)` | 对数 | `z = ln(x)` |
| `neg(x)` | 取负 | `z = -x` |

#### 归约运算

| 原语 | 功能 | 数学定义 |
|------|------|---------|
| `sum(x, axis, keepdim)` | 求和 | `z = Σ x` |
| `max(x, axis, keepdim)` | 最大值 | `z = max(x)` |
| `min(x, axis, keepdim)` | 最小值 | `z = min(x)` |

#### 线性代数

| 原语 | 功能 | 数学定义 |
|------|------|---------|
| `matmul(a, b)` | 矩阵乘 | `C = A @ B` |
| `batch_matmul(a, b)` | 批量矩阵乘 | `C[i] = A[i] @ B[i]` |

#### 内存操作

| 原语 | 功能 | 描述 |
|------|------|------|
| `load(tensor, offsets, sizes)` | 加载 | Tensor → Tile |
| `store(tile, tensor, offsets)` | 存储 | Tile → Tensor |
| `slice(tensor, offsets, sizes)` | 切片 | 子张量视图 |
| `assemble(tensor, tile, offsets)` | 组装 | Tile → Tensor |

#### 类型/布局

| 原语 | 功能 | 描述 |
|------|------|------|
| `cast(x, dtype)` | 类型转换 | 改变数据类型 |
| `reshape(x, shape)` | 重塑 | 改变形状 |
| `broadcast(x, shape)` | 广播 | 扩展形状 |

---

## 🔬 第 3 步：示例分析

### 示例：Softmax 实现

**数学公式**：
```
output[r, c] = exp(x[r, c] - max_row(x)) / sum_row(exp(x[r, c] - max_row(x)))
```

**PyPTO-Lib 实现**：

```python
@pl.program
class SoftmaxProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def softmax(self, x: pl.Tensor[[rows, cols], pl.FP32], 
                      y: pl.Out[pl.Tensor[[rows, cols], pl.FP32]]) -> pl.Tensor:
        with pl.auto_incore():
            # 行分块并行
            for r in pl.parallel(0, rows, row_chunk, chunk=1):
                tile_x = pl.slice(x, [row_chunk, cols], [r, 0])
                
                # Step 1: 行最大值（数值稳定性）
                row_max = pl.row_max(tile_x)
                
                # Step 2: 减去行最大值
                shifted = pl.row_expand_sub(tile_x, row_max)
                
                # Step 3: 指数
                exp_shifted = pl.exp(shifted)
                
                # Step 4: 行求和
                row_sum = pl.row_sum(exp_shifted)
                
                # Step 5: 行归一化
                result = pl.row_expand_div(exp_shifted, row_sum)
                
                y = pl.assemble(y, result, [r, 0])
        
        return y
```

**关键概念**：

1. **`pl.auto_incore()`**：自动标记 incore 区域
2. **`pl.parallel(0, rows, row_chunk, chunk=1)`**：行分块并行
3. **`pl.slice(x, [row_chunk, cols], [r, 0])`**：切片操作
4. **`pl.row_max(tile_x)`**：行归约（最大值）
5. **`pl.row_expand_sub(tile_x, row_max)`**：行广播减
6. **`pl.assemble(y, result, [r, 0])`**：组装结果

---

## 📊 第 4 步：编译流程

### 从原语到 PTO-ISA

```
Tensor-Level Primitives (PyPTO-Lib)
    ↓ Pass 1: Tiling
Tile-Level Operations (PTO-ISA)
    ↓ Pass 2: Memory Planning
Load/Store + Compute
    ↓ Pass 3: Sync Insertion
Synchronized PTO-ISA
    ↓ Pass 4: Code Generation
Executable Binary
```

### 关键 Pass

| Pass | 功能 | 输入 | 输出 |
|------|------|------|------|
| **Tiling** | 将 Tensor 操作分解为 Tile 循环 | Tensor IR | Tile IR |
| **Memory Planning** | 决定数据放置和移动 | Tile IR | Load/Store IR |
| **Sync Insertion** | 自动插入同步 | Load/Store IR | Synced IR |
| **Code Generation** | 生成可执行代码 | Synced IR | Binary |

---

## 🎯 第 5 步：设计原则

### 1. Tensor 级原语

**为什么不是 Tile 级？**

- Tile 级操作已经是 PTO-ISA 指令
- Tensor 级原语提供更高抽象
- 编译器负责 Tiling 优化

### 2. 硬件无关

**不指定**：
- 哪些操作在 AICore（incore）
- 哪些操作在 AICPU（orchestration）
- 如何映射到硬件单元

**由谁决定**：
- pypto backend 和 codegen
- ptoas lowering 和 target-specific codegen
- runtime（如 simpler）

### 3. Incore Scope 语义

**核心思想**：
- 用户标记计算区域
- 编译器推导参数类型
- 运行时分配 output 内存

---

## ✅ 学习总结

### 核心收获

1. **PyPTO-Lib 是张量级原语库**，类似 PyTorch ATen
2. **核心概念**：
   - Tensor vs Tile：通过 cast 转换，无数据移动
   - Incore Scope：标记计算区域，编译器推导参数
   - 参数类型：input（只读）、inout（读写）、output（运行时分配）
3. **原语集**：逐元素、归约、线性代数、内存操作、类型/布局
4. **编译流程**：Tensor → Tile → Load/Store → Sync → Binary

### 与 pypto 和 PTOAS 的关系

```
PyPTO-Lib (原语库) → pypto IR → PTOAS 编译 → pto-isa 指令 → NPU 执行
```

### 下一步学习

- **simpler**：运行时框架，学习任务调度

---

**学习状态**: ✅ 完成  
**下一步**: 进入第 5 个仓库（simpler）
