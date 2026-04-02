# pto-isa 深度学习笔记

**学习时间**: 2026-04-02  
**仓库来源**: github.com/hw-native-sys/pto-isa  
**学习状态**: ✅ 完成

---

## 📋 第 1 步：理解定位

### 核心定位

**PTO Tile Library** = PTO 指令集架构实现

- **90+ 标准 Tile 级操作**
- **跨平台支持**：Ascend A2/A3/A5/CPU
- **高性能内核**：GEMM、Flash Attention

### 在整个系统中的角色

```
PyPTO (Python DSL)  →  PTOAS (编译器)  →  pto-isa (指令集库) ← 当前学习
                                                    ↓
                                              硬件执行 (NPU)
```

### 主要职责

| 职责 | 描述 | 实现位置 |
|------|------|---------|
| **指令定义** | 90+ Tile 级操作 | `docs/isa/*.md` |
| **C++ API** | 提供类型安全的 C++ 接口 | `include/pto/common/pto_instr.hpp` |
| **跨平台实现** | 支持 A2/A3/A5/CPU | `include/pto/a2a3/`, `include/pto/a5/` |
| **高性能内核** | GEMM、Flash Attention 示例 | `kernels/manual/` |

---

## 📚 第 2 步：指令分类

### 指令总数：90+

| 类别 | 数量 | 代表指令 | 描述 |
|------|------|---------|------|
| **同步** | 1 | `TSYNC` | 同步操作（事件等待/屏障） |
| **资源绑定** | 3 | `TASSIGN`, `TSETFMATRIX` | 手动资源分配 |
| **逐元素运算** | 30+ | `TADD`, `TMUL`, `TEXP` | Tile-Tile 或 Tile-Scalar |
| **归约/扩展** | 20+ | `TCOLSUM`, `TROWMAX` | 行/列归约和广播 |
| **内存操作** | 6 | `TLOAD`, `TSTORE`, `MGATHER` | GM ↔ Tile 数据传输 |
| **矩阵乘** | 8 | `TMATMUL`, `TGEMV` | GEMM/GEMV 及变体 |
| **数据移动** | 12 | `TEXTRACT`, `TINSERT`, `TMOV` | Tile 内部数据重组 |
| **复杂操作** | 10+ | `TSORT32`, `TMRGSORT` | 排序、打印等 |
| **通信** | 9 | `TPUT`, `TGET`, `TGATHER` | 跨 NPU 通信 |

---

## 🔬 第 3 步：核心指令详解

### 核心 1: TMATMUL（矩阵乘）

**功能**：矩阵乘法（GEMM）

**数学定义**：
```
C[i,j] = Σ(k=0..K-1) A[i,k] * B[k,j]
```

**语法**：
```cpp
// C++ API
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events);

// PTO IR (DPS)
pto.tmatmul ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) 
            outs(%c : !pto.tile_buf<...>)
```

**约束**：
- **数据类型**：
  - `(int32, int8, int8)` - 量化
  - `(float, half, half)` - 混合精度
  - `(float, float, float)` - 全精度
  - `(float, bfloat16, bfloat16)` - BF16
- **Tile 位置**：
  - A: `Left`（左缓冲区）
  - B: `Right`（右缓冲区）
  - C: `Acc`（累加器）
- **形状约束**：
  - `A.rows == C.rows`
  - `A.cols == B.rows`
  - `B.cols == C.cols`
- **运行时约束**：
  - `m/k/n` ∈ `[1, 4095]`

**性能数据**（A3, 24 cores, fp16→fp32）：

| M=K=N | TMATMUL 占比 | 执行时间 |
|-------|-------------|---------|
| 1536 | 54.5% | 0.0388 ms |
| 3072 | 79.0% | 0.2067 ms |
| 6144 | 86.7% | 1.5060 ms |
| 7680 | 80.6% | 3.1680 ms |

---

### 核心 2: TLOAD（加载）

**功能**：从全局内存加载到 Tile

**数学定义**：
```
dst[i,j] = src[r0+i, c0+j]
```

**语法**：
```cpp
// C++ API
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);

// PTO IR (DPS)
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) 
          outs(%dst : !pto.tile_buf<...>)
```

**约束**：
- **数据类型**：`int8/uint8/int16/uint16/int32/uint32/int64/uint64/half/bfloat16/float`
- **Tile 位置**：`Vec` 或 `Mat`
- **布局支持**：
  - ND→ND（行主序）
  - DN→DN（列主序）
  - NZ→NZ（分形布局）

---

### 核心 3: TSYNC（同步）

**功能**：同步 PTO 执行

**两种形式**：
1. **事件等待**：`TSYNC(events...)` - 等待事件完成
2. **操作屏障**：`TSYNC<Op>()` - 为单个操作类插入屏障

**语法**：
```cpp
// C++ API - 事件等待
template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents &... events);

// C++ API - 操作屏障
template <Op OpCode>
PTO_INST void TSYNC();

// PTO IR (DPS)
pto.record_event[src_op, dst_op, eventID]
pto.wait_event[src_op, dst_op, eventID]
pto.barrier(op)
```

**支持的操作**：
- `TLOAD`, `TSTORE_ACC`, `TSTORE_VEC`
- `TMOV_M2L`, `TMOV_M2S`, `TMOV_M2B`, `TMOV_M2V`, `TMOV_V2M`
- `TMATMUL`, `TVEC`

**关键理解**：
- 前端内核**不应**手动插入同步
- 依赖 `ptoas --enable-insert-sync` 自动插入

---

## 📊 第 4 步：指令分类详解

### 类别 1: 逐元素运算（30+ 指令）

#### Tile-Tile 运算

| 指令 | 功能 | 数学定义 |
|------|------|---------|
| `TADD` | 加法 | `C = A + B` |
| `TSUB` | 减法 | `C = A - B` |
| `TMUL` | 乘法 | `C = A * B` |
| `TDIV` | 除法 | `C = A / B` |
| `TEXP` | 指数 | `C = exp(A)` |
| `TLOG` | 对数 | `C = log(A)` |
| `TRELU` | ReLU | `C = max(0, A)` |
| `TCMP` | 比较 | `C = (A op B)` |

#### Tile-Scalar 运算

| 指令 | 功能 | 数学定义 |
|------|------|---------|
| `TADDS` | 加标量 | `C = A + s` |
| `TMULS` | 乘标量 | `C = A * s` |
| `TDIVS` | 除标量 | `C = A / s` |
| `TEXPANDS` | 广播标量 | `C = broadcast(s)` |

---

### 类别 2: 归约/扩展（20+ 指令）

#### 行归约

| 指令 | 功能 | 数学定义 |
|------|------|---------|
| `TROWSUM` | 行求和 | `C[i] = Σ_j A[i,j]` |
| `TROWMAX` | 行最大值 | `C[i] = max_j A[i,j]` |
| `TROWARGMAX` | 行最大索引 | `C[i] = argmax_j A[i,j]` |

#### 列归约

| 指令 | 功能 | 数学定义 |
|------|------|---------|
| `TCOLSUM` | 列求和 | `C[j] = Σ_i A[i,j]` |
| `TCOLMAX` | 列最大值 | `C[j] = max_i A[i,j]` |
| `TCOLARGMAX` | 列最大索引 | `C[j] = argmax_i A[i,j]` |

#### 扩展（广播）

| 指令 | 功能 | 数学定义 |
|------|------|---------|
| `TROWEXPAND` | 行广播 | `C[i,:] = A[i,0]` |
| `TCOLEXPAND` | 列广播 | `C[:,j] = A[0,j]` |
| `TROWEXPANDDIV` | 行广播除 | `C[i,:] = A[i,:] / B[i]` |

---

### 类别 3: 内存操作（6 指令）

| 指令 | 功能 | 方向 |
|------|------|------|
| `TLOAD` | 加载 | GM → Tile |
| `TSTORE` | 存储 | Tile → GM |
| `TPREFETCH` | 预取 | GM → Cache |
| `MGATHER` | 收集加载 | GM → Tile（索引加载） |
| `MSCATTER` | 散射存储 | Tile → GM（索引存储） |

---

### 类别 4: 矩阵乘（8 指令）

| 指令 | 功能 | 特点 |
|------|------|------|
| `TMATMUL` | 矩阵乘 | 基础 GEMM |
| `TMATMUL_ACC` | 累加矩阵乘 | C += A @ B |
| `TMATMUL_BIAS` | 带偏置矩阵乘 | C = A @ B + bias |
| `TMATMUL_MX` | 混合精度矩阵乘 | 支持量化 |
| `TGEMV` | 矩阵向量乘 | M×K @ K×1 |
| `TGEMV_ACC` | 累加矩阵向量乘 | V += M @ V |
| `TGEMV_BIAS` | 带偏置矩阵向量乘 | V = M @ V + bias |
| `TGEMV_MX` | 混合精度矩阵向量乘 | 支持量化 |

---

### 类别 5: 数据移动（12 指令）

| 指令 | 功能 | 描述 |
|------|------|------|
| `TEXTRACT` | 提取子 Tile | 从源 Tile 提取子块 |
| `TINSERT` | 插入子 Tile | 将子块插入目标 Tile |
| `TMOV` | 移动/复制 | Tile 间数据复制 |
| `TTRANS` | 转置 | 矩阵转置 |
| `TRESHAPE` | 重塑 | 改变 Tile 形状 |
| `TFILLPAD` | 填充 | 填充无效区域 |

---

### 类别 6: 通信（9 指令）

| 指令 | 功能 | 描述 |
|------|------|------|
| `TPUT` | 远程写 | 本地 GM → 远程 GM |
| `TGET` | 远程读 | 远程 GM → 本地 GM |
| `TPUT_ASYNC` | 异步远程写 | 非阻塞 |
| `TGET_ASYNC` | 异步远程读 | 非阻塞 |
| `TNOTIFY` | 通知 | 发送标志到远程 NPU |
| `TWAIT` | 等待 | 阻塞等待信号 |
| `TTEST` | 测试 | 非阻塞测试信号 |
| `TGATHER` | 收集 | 从所有 rank 收集数据 |
| `TSCATTER` | 散射 | 向所有 rank 散射数据 |

---

## 🎯 第 5 步：核心概念

### 1. Tile 类型系统

| Tile 类型 | 位置 | 描述 | 示例 |
|----------|------|------|------|
| `Vec` | UB | 向量缓冲区 | `!pto.tile_buf<loc=vec, ...>` |
| `Mat` | UB | 矩阵缓冲区 | `!pto.tile_buf<loc=mat, ...>` |
| `Left` | L1 | 左缓冲区（矩阵乘 A） | `!pto.tile_buf<loc=left, ...>` |
| `Right` | L1 | 右缓冲区（矩阵乘 B） | `!pto.tile_buf<loc=right, ...>` |
| `Acc` | Acc | 累加器（矩阵乘 C） | `!pto.tile_buf<loc=acc, ...>` |

---

### 2. 数据类型支持

| 数据类型 | 大小 | 用途 |
|---------|------|------|
| `int8/uint8` | 1 byte | 量化计算 |
| `int16/uint16` | 2 bytes | 中间计算 |
| `int32/uint32` | 4 bytes | 累加器 |
| `int64/uint64` | 8 bytes | 大整数 |
| `half` | 2 bytes | FP16 |
| `bfloat16` | 2 bytes | BF16 |
| `float` | 4 bytes | FP32 |

---

### 3. 布局系统

| 布局 | 描述 | 用途 |
|------|------|------|
| `ND` | 行主序 | 常规布局 |
| `DN` | 列主序 | 转置布局 |
| `NZ` | 分形布局 | 矩阵乘优化 |

---

## ✅ 学习总结

### 核心收获

1. **PTO ISA 定义了 90+ Tile 级操作**，覆盖计算、内存、同步、通信等
2. **核心指令**：
   - `TMATMUL`：矩阵乘（性能关键）
   - `TLOAD/TSTORE`：内存传输
   - `TSYNC`：同步控制
3. **跨平台支持**：A2/A3/A5/CPU
4. **高性能内核**：GEMM、Flash Attention

### 与 PTOAS 的关系

```
PTOAS 编译器 → 生成 PTO IR → 调用 pto-isa 指令 → 硬件执行
```

### 下一步学习

- **pypto**：Python DSL，学习如何用 Python 编写 PTO 程序
- **pypto-lib**：算子库，学习具体算子实现
- **simpler**：运行时框架，学习任务调度

---

**学习状态**: ✅ 完成  
**下一步**: 进入第 3 个仓库（pypto）
