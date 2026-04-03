# agentlink 消息模式与 Handoff 指南

> 对齐今日 Trending 项目中“多 Agent 协作”“异步流式响应”“工具桥接”三类最佳实践。

## 什么时候需要这份文档

当系统从单 Agent 走向多 Agent 后，消息协议的难点通常不在“能不能发”，而在：

- 如何保证一次请求能被追踪
- 如何做 handoff 而不丢上下文
- 如何把同步工具调用和异步流式事件统一起来
- 如何处理超时、重试、重复消费

## 推荐的 4 种基础模式

### 1. Request / Reply

适合：一个 agent 向另一个 agent 请求明确结果。

关键字段建议：

- `message_id`
- `correlation_id`
- `sender`
- `receiver`
- `timeout_ms`
- `reply_to`

### 2. Publish / Subscribe

适合：状态广播、工具结果通知、审计事件分发。

建议额外加入：

- `topic`
- `event_type`
- `trace_id`
- `version`

### 3. Async Streaming

适合：长任务分阶段回传中间结果。

推荐事件拆分：

- `started`
- `progress`
- `partial_output`
- `completed`
- `failed`

如果项目中已经提供 `AsyncAgentBus`，建议把它作为长任务默认通道，而不是把所有输出压成一次性回复。

### 4. MCP Tool Bridge

适合：把外部工具能力接入消息总线。

建议桥接层统一完成：

- 工具发现
- 参数校验
- 执行耗时记录
- 错误规范化
- 工具输出裁剪

## Handoff 最佳实践

### 必带上下文字段

一次 handoff 至少要携带：

- 用户原始目标
- 已完成步骤
- 当前约束
- 未决问题
- 可复用中间结果

### 不要只传自然语言总结

如果只传一句“请继续处理”，下游 agent 很难判断：

- 哪些结果已验证
- 哪些工具已调用
- 哪些失败需要避开

建议 handoff payload 至少分为：

- `task`
- `state`
- `artifacts`
- `risks`
- `next_actions`

## 可靠性建议

### 幂等性

对同一个 `message_id`，消费端应该能识别重复消息。

### 超时与重试

建议把超时和重试从业务逻辑中抽离成协议层配置：

- 首次超时
- 最大重试次数
- 退避策略
- 是否允许降级

### 可观测性

至少记录：

- 排队时间
- 处理时间
- 失败原因
- 重试次数
- 最终路由路径

## 推荐新增示例

建议后续补一个完整示例，展示：

1. planner 生成任务
2. researcher 流式返回进度
3. writer 接收 handoff 并产出报告
4. MCP bridge 调用外部工具
5. trace_id 串起全链路日志

## 今日可继续补强的方向

- 增加 `docs/protocol-schema.md`
- 增加异步重试与去重示例
- 增加 trace / correlation 字段说明
- 增加 MCP 适配层错误码对照表
