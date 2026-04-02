# pypto 深度学习笔记

**学习时间**: 2026-04-02  
**仓库来源**: github.com/hw-native-sys/pypto  
**学习状态**: ✅ 完成

---

## 📋 第 1 步：理解定位

### 核心定位

**PyPTO** = Python 编程框架

- **Tile-based 编程模型**
- **多层 IR 转换**：Tensor Graph → Tile Graph → Block Graph → Execution Graph
- **自动代码生成**

### 在整个系统中的角色

```
PyPTO (Python DSL) ← 当前学习
    ↓
PTOAS (编译器)
    ↓
pto-isa (指令集库)
    ↓
硬件执行 (NPU)
```

### 主要职责

| 职责 | 描述 | 实现位置 |
|------|------|---------|
| **Python API** | 提供 Python 编程接口 | `python/pypto/` |
| **IR 构建** | 构建 Tensor/Tile/Block IR | `python/pypto/ir/` |
| **编译流程** | 多层 IR 转换和优化 | `python/pypto/ir/pass_manager.py` |
| **代码生成** | 生成 PTO IR | `python/pypto/debug/torch_codegen.py` |

---

## 📚 第 2 步：核心概念

### 1. 程序结构

```python
@pl.program
class HelloWorldProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(self, a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]) -> pl.Tensor:
        # InCore 函数：Tile 级计算
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, tile_b)
        return pl.store(tile_c, [0, 0], c)
    
    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(self, a: pl.Tensor, b: pl.Tensor, out_c: pl.Out[pl.Tensor]) -> pl.Tensor:
        # Orchestration 函数：调用 InCore 函数
        return self.tile_add(a, b, out_c)
```

**关键概念**：
- **`@pl.program`**：程序装饰器，标记一个类为 PyPTO 程序
- **`@pl.function`**：函数装饰器，有两种类型：
  - **InCore**：Tile 级计算，在单个核心上执行
  - **Orchestration**：编排函数，调用 InCore 函数
- **`pl.Tensor`**：全局内存张量
- **`pl.Tile`**：片上缓冲区
- **`pl.Out`**：输出参数标记

---

### 2. 内存层次

```
GM (Global Memory)        - 全局内存（DDR）
    ↓ pl.load
Mat (L1 Buffer)           - L1 缓冲区
    ↓ pl.move
Left/Right (L0A/L0B)      - L0 缓冲区（矩阵乘输入）
    ↓ pl.matmul
Acc (L0C)                 - 累加器（矩阵乘输出）
    ↓ pl.store
GM (Global Memory)        - 写回全局内存
```

**内存空间**：
- `pl.MemorySpace.Mat` - L1 缓冲区
- `pl.MemorySpace.Left` - L0A（矩阵乘 A）
- `pl.MemorySpace.Right` - L0B（矩阵乘 B）
- `pl.MemorySpace.Vec` - 向量缓冲区
- `pl.MemorySpace.Acc` - 累加器

---

### 3. 数据类型

| PyPTO 类型 | 描述 | 对应 C++ 类型 |
|-----------|------|--------------|
| `pl.FP32` | 32 位浮点 | `float` |
| `pl.FP16` | 16 位浮点 | `half` |
| `pl.BF16` | BF16 | `bfloat16_t` |
| `pl.INT8` | 8 位整数 | `int8_t` |
| `pl.INT32` | 32 位整数 | `int32_t` |

---

## 🔬 第 3 步：核心操作

### 1. 内存操作

#### pl.load - 加载

```python
# 从全局内存加载到 Tile
tile_a = pl.load(a, [0, 0], [128, 128])  # 偏移 [0,0]，形状 [128,128]

# 指定目标内存空间
tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
```

#### pl.store - 存储

```python
# 从 Tile 存储到全局内存
out_c = pl.store(tile_c, [0, 0], c)  # 偏移 [0,0]，目标张量 c
```

#### pl.move - 移动

```python
# 在不同内存空间之间移动
tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
```

---

### 2. 计算操作

#### 逐元素运算

```python
# 加法
tile_c = pl.add(tile_a, tile_b)

# 乘法
tile_c = pl.mul(tile_a, tile_b)

# 指数
tile_exp = pl.exp(tile_a)

# ReLU
tile_relu = pl.relu(tile_a)
```

#### 矩阵乘

```python
# 矩阵乘（初始化累加器）
tile_c = pl.matmul(tile_a, tile_b)

# 矩阵乘累加
tile_c = pl.matmul_acc(tile_c, tile_a, tile_b)
```

---

### 3. 归约操作

#### 行归约

```python
# 行最大值
row_max = pl.row_max(tile_a, max_tmp)  # shape: [64, 1]

# 行求和
row_sum = pl.row_sum(tile_a, sum_tmp)  # shape: [64, 1]
```

#### 行广播

```python
# 行广播减
shifted = pl.row_expand_sub(tile_a, row_max)  # tile_a[i,:] - row_max[i]

# 行广播除
result = pl.row_expand_div(exp_shifted, row_sum)  # exp_shifted[i,:] / row_sum[i]
```

---

## 📊 第 4 步：示例分析

### 示例 1: Hello World（元素级加法）

```python
@pl.program
class HelloWorldProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(self, a: pl.Tensor[[128, 128], pl.FP32], 
                       b: pl.Tensor[[128, 128], pl.FP32], 
                       c: pl.Out[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor:
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, tile_b)
        return pl.store(tile_c, [0, 0], c)
```

**流程**：
1. 从全局内存加载 `a` 和 `b` 到 Tile
2. 执行逐元素加法
3. 将结果存储回全局内存

---

### 示例 2: 矩阵乘（展示内存层次）

```python
@pl.program
class MatmulProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def matmul(self, a: pl.Tensor[[64, 64], pl.FP32], 
                     b: pl.Tensor[[64, 64], pl.FP32], 
                     c: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor:
        # GM → L1
        tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        
        # L1 → L0A/L0B
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        
        # 矩阵乘（L0A @ L0B → L0C）
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        
        # L0C → GM
        return pl.store(tile_c_l0c, [0, 0], c)
```

**内存流程**：
```
GM → Mat (L1) → Left/Right (L0A/L0B) → matmul → Acc (L0C) → GM
```

---

### 示例 3: Softmax（行归约 + 广播）

```python
@pl.program
class TileSoftmaxProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_softmax(self, a: pl.Tensor[[64, 64], pl.FP32], 
                           output: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor:
        tile_a = pl.load(a, [0, 0], [64, 64])
        
        # Step 1: 行最大值（数值稳定性）
        max_tmp = pl.create_tile([64, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_max = pl.row_max(tile_a, max_tmp)
        
        # Step 2: 减去行最大值
        shifted = pl.row_expand_sub(tile_a, row_max)
        
        # Step 3: 指数
        exp_shifted = pl.exp(shifted)
        
        # Step 4: 行求和
        sum_tmp = pl.create_tile([64, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum = pl.row_sum(exp_shifted, sum_tmp)
        
        # Step 5: 行归一化
        result = pl.row_expand_div(exp_shifted, row_sum)
        
        return pl.store(result, [0, 0], output)
```

**Softmax 公式**：
```
output[i] = exp(a[i] - max(a[i])) / sum(exp(a[i] - max(a[i])))
```

---

## 🎯 第 5 步：编译流程

### 多层 IR 转换

```
Tensor Graph (用户编写)
    ↓ Pass 1: Tile Partitioning
Tile Graph (Tile 级操作)
    ↓ Pass 2: Block Partitioning
Block Graph (Block 级操作)
    ↓ Pass 3: Code Generation
Execution Graph (PTO IR)
    ↓ Pass 4: PTOAS Compilation
机器码 (NPU 执行)
```

### 关键 Pass

| Pass | 功能 | 输入 | 输出 |
|------|------|------|------|
| **Tile Partitioning** | 将 Tensor 分割为 Tile | Tensor Graph | Tile Graph |
| **Block Partitioning** | 将 Tile 分配到 Block | Tile Graph | Block Graph |
| **Code Generation** | 生成 PTO IR | Block Graph | Execution Graph |

---

## ✅ 学习总结

### 核心收获

1. **PyPTO 提供 Python DSL**，让算法开发者用 Python 编写 Tile 级程序
2. **核心概念**：
   - `@pl.program` / `@pl.function` - 程序和函数装饰器
   - `pl.Tensor` / `pl.Tile` - 全局内存和片上缓冲区
   - `pl.load` / `pl.store` / `pl.move` - 内存操作
   - `pl.matmul` / `pl.add` / `pl.exp` - 计算操作
   - `pl.row_max` / `pl.row_sum` / `pl.row_expand_*` - 归约和广播
3. **内存层次**：GM → L1 → L0A/L0B → Acc → GM
4. **编译流程**：Tensor Graph → Tile Graph → Block Graph → Execution Graph

### 与 PTOAS 和 pto-isa 的关系

```
PyPTO (Python DSL) → 生成 PTO IR → PTOAS 编译 → pto-isa 指令 → NPU 执行
```

### 下一步学习

- **pypto-lib**：算子库，学习具体算子实现
- **simpler**：运行时框架，学习任务调度

---

**学习状态**: ✅ 完成  
**下一步**: 进入第 4 个仓库（pypto-lib）
