# agentlink 🔗

**AI 智能体的通信协议 —— 就像 HTTP 之于 Web，agentlink 是多智能体系统的消息总线。**

框架无关、类型安全、生产可靠。让任意两个 AI 智能体之间互相通信，无论它们使用什么底层框架。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)

[English](./README.md) | **中文**

---

## 问题背景

构建多智能体系统时，通信层是最大的麻烦：

- LangChain 智能体无法直接和 CrewAI 智能体对话
- 自己实现消息路由既脆弱又难以维护
- 消息丢失、顺序错乱、无重试机制
- 没有类型约束，接口对接全靠约定

**agentlink 是通用的智能体间通信层。** 一套 API，让任何框架的智能体都能可靠地互相通信，支持发布/订阅、请求-应答、点对点三种模式。

---

## 功能特性

- 🔀 **框架无关** — LangChain、CrewAI、AutoGen 或自定义智能体，全部兼容
- 📨 **三种通信模式** — 发布/订阅、请求-应答、点对点直发
- 🔒 **类型安全** — 基于 Pydantic 的消息 Schema 验证，接口对接不出错
- 📬 **死信队列** — 消息处理失败自动入 DLQ，支持手动重试
- 🔍 **消息追踪** — 每条消息携带 trace_id，全链路可观测
- 💾 **持久化** — SQLite/Redis 后端，智能体重启后消息不丢失
- 🌐 **服务发现** — 内置 Agent 注册与发现，支持动态扩缩容

---

## 安装

```bash
pip install agentlink
```

持久化后端（可选）：
```bash
pip install agentlink[redis]   # Redis 持久化
pip install agentlink[sqlite]  # SQLite 持久化（默认）
```

---

## 快速上手

```python
from agentlink import AgentLink

link = AgentLink()

# 智能体 A：发布消息
await link.publish("research.done", {
    "topic": "AI安全",
    "summary": "2026年主要进展...",
    "sources": ["arxiv.org/xxx"]
})

# 智能体 B：订阅消息
@link.subscribe("research.done")
async def on_research(msg):
    print(f"收到研究结果: {msg.payload['summary']}")
    # 开始写作任务...
    await link.publish("writing.start", msg.payload)
```

---

## 三种通信模式

### 发布/订阅（Pub/Sub）

适合广播事件、解耦智能体：

```python
# 多个智能体订阅同一话题
@link.subscribe("data.updated")
async def analyzer(msg): ...

@link.subscribe("data.updated")
async def notifier(msg): ...

# 发布一次，所有订阅者收到
await link.publish("data.updated", {"rows": 1000})
```

### 请求-应答（Request-Reply）

适合需要等待结果的同步调用：

```python
# 发送请求并等待回复
response = await link.request(
    recipient="calculator-agent",
    payload={"expression": "sqrt(144)"},
    timeout=30
)
print(response.result)  # → 12
```

### 点对点（Point-to-Point）

直接发送给指定智能体：

```python
await link.send("writer-agent", {
    "type": "TASK",
    "content": "请总结以下研究报告",
    "data": research_result,
    "reply_to": "coordinator"
})
```

---

## 类型安全的消息 Schema

```python
from agentlink import MessageSchema
from pydantic import BaseModel

class ResearchTask(MessageSchema):
    task_id: str
    topic: str
    depth: int = 3
    language: str = "zh"

# 发送时自动验证
await link.send("researcher", ResearchTask(
    task_id="t-001",
    topic="量子计算",
    depth=5
))

# 接收时自动解析为类型安全对象
@link.on(ResearchTask)
async def handle_task(task: ResearchTask):
    print(task.topic)   # 类型提示完整，IDE 自动补全
    print(task.depth)   # int，不是 str
```

---

## 死信队列（DLQ）

生产环境必备，消息处理失败不丢失：

```python
link = AgentLink(
    dlq_enabled=True,
    max_retries=3,
    retry_backoff="exponential"  # 1s → 2s → 4s
)

# 查看失败消息
for msg in link.dlq.all():
    print(f"失败: {msg.id} | 原因: {msg.last_error}")
    link.dlq.retry(msg)  # 手动重试
```

---

## 持久化订阅

智能体重启后自动恢复订阅，不丢失中间消息：

```python
link = AgentLink(
    persistence=PersistenceConfig(
        backend="sqlite",              # 或 "redis"
        path="./messages.db",
        subscription_ttl=86400        # 订阅保持24小时
    )
)

# 重启后自动恢复
await link.connect(agent_id="worker-1", auto_resume=True)
# → 自动接收重启期间积压的消息
```

---

## 消息追踪与可观测性

```python
# 发送时附带 trace_id
await link.send("agent-b", payload, trace_id="req-abc123")

# 查询完整消息链路
trace = link.tracer.get_trace("req-abc123")
# AgentA (0ms) → router (2ms) → AgentB (5ms) → reply (48ms)

# 集成 OpenTelemetry
link.enable_otel(endpoint="http://jaeger:4317")
```

---

## 框架集成

### LangChain

```python
from agentlink.adapters.langchain import LangChainAgentAdapter

lc_adapter = LangChainAgentAdapter(
    agent=langchain_agent,
    link=link,
    agent_id="lc-researcher"
)
# LangChain 智能体现在可以接收和发送 agentlink 消息
```

### CrewAI

```python
from agentlink.adapters.crewai import CrewAIAdapter

crew_adapter = CrewAIAdapter(crew_agent, link=link)
```

### 自定义智能体

```python
class MyAgent:
    def __init__(self):
        self.link = AgentLink()
        
    async def start(self):
        self.link.register(agent_id="my-agent")
        
        @self.link.on_message
        async def handle(msg):
            result = await self.process(msg.content)
            await self.link.reply(msg, result)
```

---

## 对比同类方案

| 功能 | agentlink | LangGraph | CrewAI | AutoGen |
|------|-----------|-----------|--------|---------|
| 框架无关 | ✅ | ❌ | ❌ | ❌ |
| 消息类型安全 | ✅ | ❌ | ❌ | 部分 |
| 死信队列 | ✅ | ❌ | ❌ | ❌ |
| MCP 兼容 | 🔄 | ❌ | ❌ | ❌ |
| 持久化 | ✅ | 部分 | ❌ | ❌ |
| 消息追踪 | ✅ | 部分 | ❌ | 部分 |

---

## 路线图

- [ ] MCP（Model Context Protocol）兼容层 —— 作为 MCP Server 暴露接口
- [ ] A2A（Agent2Agent）v0.3 协议适配
- [ ] 服务发现（Consul/Etcd 后端）
- [ ] 消息加密传输
- [ ] Kubernetes 部署 Helm Chart
- [ ] 监控仪表盘（Grafana 模板）

---

## 示例

```
examples/
  01_simple_chat.py           # 两个智能体互相通信
  02_request_reply.py         # 同步请求-应答模式
  03_pubsub_pipeline.py       # 发布订阅流水线
  04_langchain_integration.py # LangChain 集成
  05_error_handling.py        # DLQ 与重试机制
  06_persistent_queue.py      # SQLite 持久化消息
```

---

## 许可证

MIT © cdzzy
