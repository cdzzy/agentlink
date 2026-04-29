# agentlink 🔗

**The inter-agent communication protocol.**

Like HTTP is to web services, AgentLink is the missing protocol layer that lets AI agents built with **different frameworks** talk to each other.

```
LangGraph Agent  ──┐
AutoGen Agent    ──┼──► AgentLink Bus ──► Any Agent
CrewAI Agent     ──┘
Your Custom Agent ─┘
```

---

## The Problem

Every AI agent framework is an island. A LangGraph agent can't talk to an AutoGen agent. A CrewAI crew can't delegate to a custom Python agent. Every "multi-agent system" is really a monolith — all agents must be written in the same framework.

This is the exact problem HTTP solved for web services in 1991. Before HTTP, every server spoke a different protocol. After HTTP, anything could talk to anything.

**AgentLink is that protocol for agents.**

---

## Installation

```bash
pip install agentlink
```

Or from source:
```bash
git clone https://github.com/cdzzy/agentlink
cd agentlink
pip install -e .
```

---

## Quick Start

```python
from agentlink import AgentNode, AgentBus, AgentMessage

# Define your agents (any callable works)
def researcher(message: AgentMessage) -> str:
    return f"Research results for: {message.content}"

def writer(message: AgentMessage) -> str:
    return f"Polished article about: {message.content}"

# Create nodes
researcher_node = AgentNode("researcher", researcher, capabilities=["web-search"])
writer_node     = AgentNode("writer",     writer,     capabilities=["writing"])

# Connect to a bus
bus = AgentBus()
bus.register_many(researcher_node, writer_node)

# Now they can talk to each other
reply = researcher_node.send("writer", "AI trends in 2026")
print(reply.content)
# → "Polished article about: AI trends in 2026"
```

---

## Core Concepts

### The Protocol: `AgentMessage`

Every inter-agent communication is an `AgentMessage` — a self-describing, serializable envelope:

```python
from agentlink.protocol.message import AgentMessage, MessageType, AgentAddress

msg = AgentMessage(
    type=MessageType.REQUEST,
    sender=AgentAddress("planner", "production"),
    recipient=AgentAddress("researcher", "production"),
    content="Find the latest AI news",
    content_type="text/plain",
    ttl=60,   # expires in 60 seconds
)

# Built-in operations
reply = msg.reply("Here are the findings...")
error = msg.error("Service unavailable")
forwarded = msg.forward_to(AgentAddress("backup-researcher", "production"))

# Serialize for transport
data = msg.to_dict()
restored = AgentMessage.from_dict(data)
```

**Message types:**

| Type | Description |
|------|-------------|
| `REQUEST` | Ask another agent to do something |
| `REPLY` | Response to a REQUEST |
| `ERROR` | Something went wrong |
| `EVENT` | Fire-and-forget notification |
| `STREAM_*` | Streaming response chunks |
| `PING/PONG` | Health check |
| `ANNOUNCE` | "I'm online with these capabilities" |
| `DELEGATE` | Transfer task ownership |
| `FORWARD` | Route to different agent |

---

### Agent Addressing: `AgentAddress`

```python
from agentlink.protocol.message import AgentAddress

# Full address
addr = AgentAddress("researcher", "production")
# → "researcher@production"

# Parse from string
addr = AgentAddress.parse("researcher@production/web-search")
# → agent_id="researcher", namespace="production", capability="web-search"

# Wildcard
addr = AgentAddress("*", "*")   # matches any agent in any namespace
```

---

### Capability Declaration

```python
from agentlink.protocol.capability import AgentCapability, WELL_KNOWN_CAPABILITIES

# Use well-known capabilities for interoperability
caps = [
    WELL_KNOWN_CAPABILITIES["web-search"],
    WELL_KNOWN_CAPABILITIES["summarize"],
]

# Or define custom ones
custom_cap = AgentCapability(
    name="financial-analysis",
    description="Analyze financial data and produce reports",
    tags=["finance", "analysis"],
    input_schema={"type": "string"},
    output_schema={"type": "object"},
)
```

**Well-known capabilities:** `web-search`, `summarize`, `code-execution`, `rag-retrieval`, `image-analysis`, `planning`, `memory`

---

### The Bus: `AgentBus`

```python
from agentlink import AgentBus

bus = AgentBus(name="production")

# Register agents
bus.register(planner_node)
bus.register(researcher_node)

# Direct bus-level send (for testing / external systems)
reply = bus.send("planner", "researcher", "Start the research task")

# Add middleware
def auth_middleware(msg):
    if msg.metadata.get("api_key") != SECRET:
        return None   # drop the message
    return msg

bus.use(auth_middleware)

# Inspect
bus.print_status()
print(bus.stats)
```

---

### Framework Adapters

#### LangGraph

```python
from agentlink.adapters import LangGraphAdapter

# compiled_graph = your_graph.compile()
node = LangGraphAdapter(
    graph=compiled_graph,
    agent_id="lg-planner",
    capabilities=["planning"],
    namespace="production",
).as_node()

bus.register(node)
```

#### AutoGen

```python
from agentlink.adapters import AutoGenAdapter

node = AutoGenAdapter(
    agent=assistant_agent,
    agent_id="ag-analyst",
    initiator=user_proxy,
    capabilities=["data-analysis", "code-execution"],
    max_turns=5,
).as_node()

bus.register(node)
```

#### CrewAI

```python
from agentlink.adapters import CrewAIAdapter

node = CrewAIAdapter(
    crew=my_crew,
    agent_id="report-crew",
    capabilities=["research", "report-writing"],
    task_input_key="topic",   # matches {topic} in your task descriptions
).as_node()

bus.register(node)
```

#### A2A Protocol:

```python
from agentlink.adapters.a2a_adapter import A2AAdapter, A2AServerAdapter

# Connect to a remote A2A agent
node = A2AAdapter(
    agent_url="http://analyst-agent:8000",
    agent_id="remote-analyst",
    capabilities=["data-analysis"],
).as_node()

bus.register(node)

# Expose your node as an A2A server (publishes Agent Card at /.well-known/agent.json)
server = A2AServerAdapter(node=my_node, host="0.0.0.0", port=8000)
# await server.start()
```

#### Any Python Callable

```python
from agentlink.adapters import GenericAdapter

node = GenericAdapter(
    fn=lambda text: f"processed: {text}",
    agent_id="simple-agent",
    capabilities=["general"],
).as_node()
```

---

### Routing Strategies

```python
# 1. Direct: send to a specific agent by ID
reply = my_node.send("researcher@production", "Find AI news")

# 2. Capability-based: send to ANY agent with this capability
reply = my_node.send("web-search", "Find AI news")  # finds first capable agent

# 3. Cross-namespace
reply = my_node.send("researcher@team-b", "Cross-team request")

# 4. Broadcast event to all agents in namespace
my_node.broadcast("Deployment complete", event_type="system_event")

# 5. Health check
alive = my_node.ping("researcher")   # returns True/False
```

---

### The Message Envelope

For transport, messages are wrapped in a `MessageEnvelope`:

```python
from agentlink.protocol.message import MessageEnvelope

envelope = MessageEnvelope(
    message=msg,
    protocol_version="agentlink/1.0",
    max_hops=10,
)

envelope.record_hop("node-1")
envelope.is_loop_detected()   # True if hop_count >= max_hops

# Serialize to JSON for network transport
import json
json.dumps(envelope.to_dict())
```

---

## Multi-Framework Example

```python
from agentlink import AgentBus
from agentlink.adapters import LangGraphAdapter, AutoGenAdapter, GenericAdapter

# Each agent from a different framework
planner  = LangGraphAdapter(lg_graph,  "planner",  capabilities=["planning"]).as_node()
analyst  = AutoGenAdapter(ag_agent, "analyst", initiator=proxy, capabilities=["analysis"]).as_node()
reporter = GenericAdapter(my_fn, "reporter", capabilities=["writing"]).as_node()

bus = AgentBus("production")
bus.register_many(planner, analyst, reporter)

# They all speak the same protocol now
result = planner.send("analyst",  "Analyze Q1 2026 AI market data")
report  = analyst.send("reporter", result.content)
print(report.content)
```

---

## Examples

```
examples/
  01_quickstart.py          # Two agents talking to each other
  02_cross_framework.py     # LangGraph + AutoGen + CrewAI on one bus
  03_capability_routing.py  # Semantic routing by capability
  04_real_integration.py    # Real LLM calls (needs API key)
```

---

## Design Principles

1. **Framework-agnostic**: The core has zero dependencies on any agent framework
2. **Zero dependencies**: Core protocol and runtime require only Python stdlib
3. **Minimal surface**: One message format, one bus, one address scheme
4. **Extensible**: Adapters, middleware, and transport layers are pluggable
5. **Observable**: Every message is logged; middleware can inspect/modify/block

---

## Comparison

| Concern | MCP | AgentLink |
|---------|-----|-----------|
| Model calls tools | ✅ | ✅ |
| Agent calls agent | ❌ | ✅ |
| Cross-framework | ❌ | ✅ |
| Capability discovery | ❌ | ✅ |
| Message correlation | ❌ | ✅ |
| Middleware pipeline | ❌ | ✅ |

MCP solves "model → tool". AgentLink solves "agent → agent".

---

## Roadmap

- [x] Async support (`async def` handlers) ✅ (examples/05_async_streaming.py)
- [x] **MCP Hub** (multi-server coordination) ✅ (agentlink/extensions/mcp_hub.py)
- [x] **A2A Protocol adapter** (Google's Agent-to-Agent protocol — server & client, inspired by a2a-protocol.org) ✅ (agentlink/adapters/a2a_adapter.py)
- [ ] Network transport (WebSocket, gRPC, Redis pub/sub)
- [ ] AgentLink Hub (distributed registry)
- [ ] Message signing & verification (trust between agents)
- [ ] OpenTelemetry tracing integration
- [ ] Stream support for long-running tasks
- [ ] CLI: `agentlink serve`, `agentlink send`, `agentlink status`

---

## Contributing

PRs welcome. The goal: be the smallest, most composable inter-agent protocol layer — the kind of thing that's obviously right in hindsight.

```bash
git clone https://github.com/cdzzy/agentlink
cd agentlink
pip install -e ".[dev]"
pytest tests/ -v
python examples/01_quickstart.py
```

---

## License

MIT
