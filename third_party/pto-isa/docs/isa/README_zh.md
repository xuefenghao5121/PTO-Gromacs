<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA 手册与参考

本文档目录是 PTO ISA 的权威文档树。它将架构手册、指令集指南、家族契约和精确的指令参考分组整合在同一个位置。

## PTO ISA 中的文本汇编

本树是权威的 PTO ISA 手册。文本汇编拼写属于 PTO ISA 的语法层，而非第二份并行的架构手册。

- PTO ISA 定义了架构可见的语义、合法性、状态、排序、目标 profile 边界，以及 `pto.t*`、`pto.v*`、`pto.*` 及其他操作的可见行为
- PTO-AS 是用于编写这些操作和操作数的汇编拼写。它是 PTO ISA 的表达方式的一部分，而非具有不同语义的分立 ISA

如果问题是"PTO 程序在 CPU、A2/A3 和 A5 上的含义是什么？"，请留在本树中。如果问题是"这个操作的操作数形状或文本拼写是什么？"，请使用本树中语法与操作数相关的页面。

## 从这里开始

## 轴归约 / 扩展
- [TROWSUM](TROWSUM_zh.md) - 通过对列求和来归约每一行。
- [TROWPROD](TROWPROD_zh.md) - 通过跨列乘积来归约每一行。
- [TCOLSUM](TCOLSUM_zh.md) - 通过对行求和来归约每一列。
- [TCOLPROD](TCOLPROD_zh.md) - 通过跨行乘积来归约每一列。
- [TCOLMAX](TCOLMAX_zh.md) - 通过取行间最大值来归约每一列。
- [TROWMAX](TROWMAX_zh.md) - 通过取列间最大值来归约每一行。
- [TROWMIN](TROWMIN_zh.md) - 通过取列间最小值来归约每一行。
- [TROWARGMAX](TROWARGMAX_zh.md) - 获取每行最大值对应列索引。
- [TROWARGMIN](TROWARGMIN_zh.md) - 获取每行最小值对应列索引。
- [TCOLARGMAX](TCOLARGMAX_zh.md) - 获取每列最大值对应行索引。
- [TCOLARGMIN](TCOLARGMIN_zh.md) - 获取每列最小值对应行索引。
- [TROWEXPAND](TROWEXPAND_zh.md) - 将每个源行的第一个元素广播到目标行中。
- [TROWEXPANDDIV](TROWEXPANDDIV_zh.md) - 行广播除法：将 `src0` 的每一行除以一个每行标量向量 `src1`。
- [TROWEXPANDMUL](TROWEXPANDMUL_zh.md) - 行广播乘法：将 `src0` 的每一行乘以一个每行标量向量 `src1`。
- [TROWEXPANDSUB](TROWEXPANDSUB_zh.md) - 行广播减法：从 `src0` 的每一行中减去一个每行标量向量 `src1`。
- [TROWEXPANDADD](TROWEXPANDADD_zh.md) - 行广播加法：加上一个每行标量向量。
- [TROWEXPANDMAX](TROWEXPANDMAX_zh.md) - 行广播最大值：与每行标量向量取最大值。
- [TROWEXPANDMIN](TROWEXPANDMIN_zh.md) - 行广播最小值：与每行标量向量取最小值。
- [TROWEXPANDEXPDIF](TROWEXPANDEXPDIF_zh.md) - 行指数差运算：计算 exp(src0 - src1)，其中 src1 为每行标量。
- [TCOLMIN](TCOLMIN_zh.md) - 通过取行间最小值来归约每一列。
- [TCOLEXPAND](TCOLEXPAND_zh.md) - 将每个源列的第一个元素广播到目标列中。
- [TCOLEXPANDDIV](TCOLEXPANDDIV_zh.md) - 列广播除法：将每一列除以一个每列标量向量。
- [TCOLEXPANDMUL](TCOLEXPANDMUL_zh.md) - 列广播乘法：将每一列乘以一个每列标量向量。
- [TCOLEXPANDADD](TCOLEXPANDADD_zh.md) - 列广播加法：对每一列加上每列标量向量。
- [TCOLEXPANDMAX](TCOLEXPANDMAX_zh.md) - 列广播最大值：与每列标量向量取最大值。
- [TCOLEXPANDMIN](TCOLEXPANDMIN_zh.md) - 列广播最小值：与每列标量向量取最小值。
- [TCOLEXPANDSUB](TCOLEXPANDSUB_zh.md) - 列广播减法：从每一列中减去一个每列标量向量。
- [TCOLEXPANDEXPDIF](TCOLEXPANDEXPDIF_zh.md) - 列指数差运算：计算 exp(src0 - src1)，其中 src1 为每列标量。

## 模型层次

阅读顺序与手册章节地图一致：先编程模型与机器模型，再语法与状态，再内存，最后是操作码参考。

- [编程模型](programming-model/tiles-and-valid-regions_zh.md)
- [机器模型](machine-model/execution-agents_zh.md)
- [语法与操作数](syntax-and-operands/assembly-model_zh.md)
- [类型系统](state-and-types/type-system_zh.md)
- [位置意图与合法性](state-and-types/location-intent-and-legality_zh.md)
- [内存模型](memory-model/consistency-baseline_zh.md)

## 复杂指令
- [TPRINT](TPRINT_zh.md) - 调试/打印 Tile 中的元素（实现定义）。
- [TMRGSORT](TMRGSORT_zh.md) - 用于多个已排序列表的归并排序（实现定义的元素格式和布局）。
- [TSORT32](TSORT32_zh.md) - 对 `src` 的每个 32 元素块，与 `idx` 中对应的索引一起进行排序，并将排序后的值-索引对写入 `dst`。
- [TGATHER](TGATHER_zh.md) - 使用索引 Tile 或编译时掩码模式来收集/选择元素。
- [TCI](TCI_zh.md) - 生成连续整数序列到目标 Tile 中。
- [TTRI](TTRI_zh.md) - 生成三角（下/上）掩码 Tile。
- [TRANDOM](TRANDOM_zh.md) - 使用基于计数器的密码算法在目标 Tile 中生成随机数。
- [TPARTADD](TPARTADD_zh.md) - 部分逐元素加法，对不匹配的有效区域具有实现定义的处理方式。
- [TPARTMUL](TPARTMUL_zh.md) - 部分逐元素乘法，对有效区域不一致的处理为实现定义。
- [TPARTMAX](TPARTMAX_zh.md) - 部分逐元素最大值，对不匹配的有效区域具有实现定义的处理方式。
- [TPARTMIN](TPARTMIN_zh.md) - 部分逐元素最小值，对不匹配的有效区域具有实现定义的处理方式。
- [TGATHERB](TGATHERB_zh.md) - 使用字节偏移量收集元素。
- [TSCATTER](TSCATTER_zh.md) - 使用逐元素行索引将源 Tile 的行散播到目标 Tile 中。
- [TQUANT](TQUANT_zh.md) - 量化 Tile（例如 FP32 到 FP8），生成指数/缩放/最大值输出。

- [指令集总览](instruction-surfaces/README_zh.md)
- [指令族](instruction-families/README_zh.md)
- [指令描述格式](reference/format-of-instruction-descriptions_zh.md)
- [Tile 指令集参考](tile/README_zh.md)
- [Vector 指令集参考](vector/README_zh.md)
- [标量与控制参考](scalar/README_zh.md)
- [其他与通信参考](other/README_zh.md)
- [通用约定](conventions_zh.md)

## 支持性参考

- [参考注释](reference/README_zh.md)（术语表、诊断、可移植性、规范来源）

## 兼容性重定向

`tile/`、`vector/`、`scalar/` 和 `other/` 下的分组指令集树是权威的 PTO ISA 路径。

部分旧的根级 tile 页面（如 `TADD_zh.md`、`TLOAD_zh.md`、`TMATMUL_zh.md` 等）现仅作为兼容性重定向保留，以避免现有链接立即失效。新 PTO ISA 文档应链接到分组指令集路径，尤其是以下位置的独立 per-op 页面：

- `docs/isa/tile/ops/`
- `docs/isa/vector/ops/`
- `docs/isa/scalar/ops/`
