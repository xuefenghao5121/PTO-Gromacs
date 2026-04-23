# 其他指令族

“其他”指令族覆盖通信、运行时和支撑性操作的共享契约。这些操作虽然是架构可见行为，但与 tile、vector、scalar/control 的主干模式不同。

## 指令族概览

| 指令族 | 说明 | Profile |
| --- | --- | --- |
| 通信与运行时 | 并行组上的点对点与集合通信 | A2/A3, A5 |
| 非 ISA 与支撑操作 | tile 序列、量化、释放与辅助操作 | 依操作而定 |

## 共享操作数模型

- 通信族使用并行组句柄、GM 视图、暂存 tile 和异步事件对象。
- 支撑族使用 tile、tile 序列、量化参数和资源句柄。

## 共享副作用

- 通信族会引入跨 NPU 的排序与可见性副作用。
- 支撑族可能改变资源状态、量化表示或 tile 序列结构。

## 共享约束

- 并行组一致性
- 缓冲区角色与尺寸匹配
- 量化参数合法
- profile 支持子集明确可见

## 不允许的情形

- 集合操作中各 rank 的协议不一致
- 依赖未声明的 backend 便利实现
- 在不支持的 profile 上使用 A5 专属支撑操作

## 相关页面

- [其他指令集](../instruction-surfaces/other-instructions_zh.md)
- [通信与运行时](../other/communication-and-runtime_zh.md)
- [非 ISA 与支撑操作](../other/non-isa-and-supporting-ops_zh.md)
