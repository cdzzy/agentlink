"""
Example 1: Quick Start
Two agents talking to each other over an AgentLink bus.

Run: python examples/01_quickstart.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlink import AgentNode, AgentBus, AgentMessage


# ── Define your agents (any callable works) ──────────────────────

def researcher_agent(message: AgentMessage) -> str:
    """Simulates a research agent."""
    query = str(message.content)
    return (
        f"[Researcher] Results for '{query}':\n"
        f"  - Finding 1: AI agents are transforming enterprise workflows in 2026.\n"
        f"  - Finding 2: Multi-agent systems outperform single agents on complex tasks.\n"
        f"  - Finding 3: MCP protocol adoption is accelerating across the industry."
    )


def writer_agent(message: AgentMessage) -> str:
    """Simulates a writing agent."""
    content = str(message.content)
    return (
        f"[Writer] Here's a polished summary:\n\n"
        f"Based on recent research: {content[:100]}...\n\n"
        f"This represents a significant shift in how AI systems collaborate."
    )


# ── Build the network ─────────────────────────────────────────────

bus = AgentBus(name="quickstart-demo")

researcher = AgentNode(
    agent_id="researcher",
    handler=researcher_agent,
    capabilities=["web-search", "research"],
    description="Finds and summarizes information",
)

writer = AgentNode(
    agent_id="writer",
    handler=writer_agent,
    capabilities=["writing", "summarize"],
    description="Produces polished written content",
)

# Register both agents on the same bus
bus.register_many(researcher, writer)

# ── They can now talk to each other ──────────────────────────────

print("\n" + "="*60)
print("  AgentLink Quickstart Demo")
print("="*60)

# Writer asks researcher for information
print("\n[1] Writer -> Researcher: requesting research...")
reply = writer.send("researcher", "Find the latest trends in AI agent frameworks")
print(f"\n{reply.content}")

# Researcher asks writer to polish its output
print("\n[2] Researcher -> Writer: requesting a polished summary...")
reply = researcher.send("writer", "AI agents are growing rapidly. LangGraph hit 15k stars.")
print(f"\n{reply.content}")

# Check if an agent is alive
print("\n[3] Ping test...")
alive = writer.ping("researcher")
print(f"  researcher is alive: {alive}")

# Show bus status
bus.print_status()
