# 其他指令集

“其他”指令集覆盖不适合归入 tile、vector 或 scalar/control 主干的可见操作，包括跨 NPU 通信和若干支撑性操作。

## 指令集概览

| 类别 | 说明 | Profile |
| --- | --- | --- |
| 通信与运行时 | 并行组上的点对点与集合通信 | A2/A3, A5 |
| 非 ISA 与支撑操作 | tile 序列、量化、释放与辅助操作 | 依操作而定 |

## 输入

这类指令集常见输入包括：

- 并行组句柄
- 本地或远端 GM 视图
- 暂存 tile
- tile 序列
- 量化参数或控制参数

## 输出

其他指令集会产生：

- 跨 rank 的数据传输结果
- 集合归约结果
- tile 序列或量化表示的变化
- 资源状态变化

## 约束

- 通信操作必须遵守并行组一致性。
- 支撑操作仍然必须遵守 PTO 的类型、形状和 profile 约束。
- 不同操作的副作用差异较大，不能套用 tile 或 vector 的默认推理方式。

## 不允许的情形

- 把 CPU 仿真路径当作跨 NPU 通信 profile
- 在不满足 profile 条件时使用 A5 专属支撑操作
- 把 tile 序列操作误当成普通逐元素指令

## 相关页面

- [通信与运行时](../other/communication-and-runtime_zh.md)
- [非 ISA 与支撑操作](../other/non-isa-and-supporting-ops_zh.md)
- [其他指令族](../instruction-families/other-families_zh.md)
