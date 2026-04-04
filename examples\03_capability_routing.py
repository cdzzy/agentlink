"""
Example 3: Capability-based routing + middleware pipeline

Demonstrates how agents can discover and route to each other
by capability rather than hard-coded IDs.

Run: python examples/03_capability_routing.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from agentlink import AgentNode, AgentBus, AgentMessage
from agentlink.protocol.capability import AgentCapability, WELL_KNOWN_CAPABILITIES
from agentlink.protocol.message import MessageType


# ── Agents ────────────────────────────────────────────────────────

def search_agent_1(message: AgentMessage) -> str:
    return f"[Search-1] Found: latest news about '{message.content}'"

def search_agent_2(message: AgentMessage) -> str:
    return f"[Search-2] Alternative results for '{message.content}'"

def rag_agent(message: AgentMessage) -> str:
    return f"[RAG] Retrieved 5 documents from knowledge base for: '{message.content}'"

def code_agent(message: AgentMessage) -> str:
    code = f"result = {len(str(message.content))} * 42"
    return f"[Code] Executed: `{code}` → {len(str(message.content)) * 42}"

def orchestrator(message: AgentMessage) -> str:
    return f"[Orchestrator] Received: {message.content}"


# ── Build bus with multiple capable agents ─────────────────────

bus = AgentBus("capability-demo")

search1 = AgentNode("search-agent-1", search_agent_1,
                    capabilities=[WELL_KNOWN_CAPABILITIES["web-search"]])
search2 = AgentNode("search-agent-2", search_agent_2,
                    capabilities=[WELL_KNOWN_CAPABILITIES["web-search"]])
rag     = AgentNode("rag-agent",      rag_agent,
                    capabilities=[WELL_KNOWN_CAPABILITIES["rag-retrieval"]])
coder   = AgentNode("code-agent",     code_agent,
                    capabilities=[WELL_KNOWN_CAPABILITIES["code-execution"]])
orch    = AgentNode("orchestrator",   orchestrator,
                    capabilities=["orchestration"])

bus.register_many(search1, search2, rag, coder, orch)

# ── Middleware: timing ─────────────────────────────────────────

timings = {}

def timing_middleware(msg: AgentMessage) -> AgentMessage:
    timings[msg.id] = time.perf_counter()
    return msg

bus.use(timing_middleware)

# ── Demo ────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  AgentLink — Capability Routing Demo")
print("="*60)

print("\n[1] Direct routing by agent ID:")
reply = orch.send("rag-agent", "What is a vector database?")
print(f"  -> {reply.content}")

print("\n[2] Capability-based routing (find any 'web-search' agent):")
reply = orch.send("web-search", "Latest AI news 2026")
print(f"  -> {reply.content}")

print("\n[3] Capability-based routing ('code-execution'):")
reply = orch.send("code-execution", "Calculate the answer")
print(f"  -> {reply.content}")

print("\n[4] Cross-namespace query:")
reply = orch.send("rag-retrieval", "Search knowledge base for: agent protocols")
print(f"  -> {reply.content}")

print("\n[5] Ping / health check:")
for agent_id in ["rag-agent", "code-agent", "search-agent-1"]:
    alive = orch.ping(agent_id)
    status = "alive" if alive else "unreachable"
    print(f"  {agent_id:20s}: {status}")

print("\n[6] Broadcast event to all agents in namespace:")
orch.broadcast("System maintenance in 5 minutes", event_type="system_notice")
print("  Broadcast sent.")

# Registry discovery
print("\n[7] Registry discovery:")
capable_of_search = bus.registry.find_by_capability("web-search")
print(f"  Agents with 'web-search': {[n.agent_id for n in capable_of_search]}")

bus.print_status()
