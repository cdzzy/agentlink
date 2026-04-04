"""
Example 2: Cross-Framework Agent Collaboration

Shows LangGraph, AutoGen, and CrewAI agents running on the same bus
using mock implementations (no real framework install needed to run this).

In production, replace the mock agents with real framework instances
and swap GenericAdapter for LangGraphAdapter / AutoGenAdapter / CrewAIAdapter.

Run: python examples/02_cross_framework.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlink import AgentNode, AgentBus, AgentMessage
from agentlink.adapters import GenericAdapter
from agentlink.protocol.capability import AgentCapability


print("\n" + "="*60)
print("  AgentLink — Cross-Framework Collaboration Demo")
print("="*60)


# ─────────────────────────────────────────────────────────────────
# Mock agents simulating different frameworks
# ─────────────────────────────────────────────────────────────────

def mock_langgraph_agent(input_text: str) -> str:
    """
    Simulates a LangGraph planning agent.
    Real usage: LangGraphAdapter(compiled_graph, agent_id="planner")
    """
    return (
        f"[LangGraph Planner] Breaking down task: '{input_text}'\n"
        f"  Step 1: Gather data via researcher agent\n"
        f"  Step 2: Analyze findings via analyst agent\n"
        f"  Step 3: Write final report via writer agent\n"
        f"  Plan ready. Delegating to researcher..."
    )


def mock_autogen_agent(input_text: str) -> str:
    """
    Simulates an AutoGen code/analysis agent.
    Real usage: AutoGenAdapter(assistant, agent_id="analyst", initiator=user_proxy)
    """
    return (
        f"[AutoGen Analyst] Analyzing: '{input_text[:60]}...'\n\n"
        f"```python\n"
        f"# Analysis results\n"
        f"findings = {{'trend': 'upward', 'confidence': 0.87, 'samples': 1200}}\n"
        f"```\n\n"
        f"Conclusion: Strong positive trend detected with 87% confidence."
    )


def mock_crewai_agent(input_text: str) -> str:
    """
    Simulates a CrewAI multi-agent crew producing a report.
    Real usage: CrewAIAdapter(crew, agent_id="report-crew")
    """
    return (
        f"[CrewAI Report Crew] Final report on: '{input_text[:50]}...'\n\n"
        f"Executive Summary:\n"
        f"  Our research team (Researcher + Analyst + Writer) has completed\n"
        f"  the analysis. Key findings indicate strong market opportunity.\n\n"
        f"  Recommendation: Proceed with implementation."
    )


def mock_custom_agent(input_text: str) -> str:
    """A plain Python agent — no framework needed."""
    return f"[Custom Agent] Orchestration complete. All agents responded successfully."


# ─────────────────────────────────────────────────────────────────
# Create adapter nodes for each "framework"
# ─────────────────────────────────────────────────────────────────

planner_node = GenericAdapter(
    fn=mock_langgraph_agent,
    agent_id="planner",
    capabilities=["planning", "task-decomposition"],
    description="LangGraph-based planning agent",
    namespace="production",
).as_node()

analyst_node = GenericAdapter(
    fn=mock_autogen_agent,
    agent_id="analyst",
    capabilities=["code-execution", "data-analysis"],
    description="AutoGen-based analysis agent",
    namespace="production",
).as_node()

report_crew_node = GenericAdapter(
    fn=mock_crewai_agent,
    agent_id="report-crew",
    capabilities=["research", "report-writing", "summarize"],
    description="CrewAI-based report generation crew",
    namespace="production",
).as_node()

orchestrator_node = GenericAdapter(
    fn=mock_custom_agent,
    agent_id="orchestrator",
    capabilities=["orchestration"],
    description="Custom Python orchestrator",
    namespace="production",
).as_node()


# ─────────────────────────────────────────────────────────────────
# Connect everything to one bus
# ─────────────────────────────────────────────────────────────────

bus = AgentBus(name="production")
bus.register_many(planner_node, analyst_node, report_crew_node, orchestrator_node)

# Add logging middleware
def log_middleware(msg: AgentMessage) -> AgentMessage:
    print(f"  [bus] {msg.type.value:10s}  {str(msg.sender):<20} -> {str(msg.recipient)}")
    return msg

bus.use(log_middleware)


# ─────────────────────────────────────────────────────────────────
# Orchestrate a multi-framework workflow
# ─────────────────────────────────────────────────────────────────

task = "Analyze the market opportunity for AI agent testing tools in 2026"

print(f"\nTask: {task}")
print("-" * 60)

# Step 1: Orchestrator asks LangGraph planner to break down the task
print("\n[Step 1] Orchestrator -> LangGraph Planner")
plan = orchestrator_node.send("planner@production", task)
print(plan.content)

# Step 2: LangGraph planner asks AutoGen analyst to run analysis
print("\n[Step 2] LangGraph Planner -> AutoGen Analyst")
analysis = planner_node.send("analyst@production", f"Analyze: {task}")
print(analysis.content)

# Step 3: AutoGen analyst asks CrewAI crew to write the report
print("\n[Step 3] AutoGen Analyst -> CrewAI Report Crew")
report = analyst_node.send("report-crew@production", f"Write report: {task}")
print(report.content)

# Step 4: Capability-based routing — find any agent that can summarize
print("\n[Step 4] Capability routing: find any agent with 'summarize' capability")
summary = orchestrator_node.send("summarize", "Summarize the workflow results")
print(summary.content)

# ─────────────────────────────────────────────────────────────────
# Show what we built
# ─────────────────────────────────────────────────────────────────
bus.print_status()
print(f"Total messages in log: {len(bus.message_log)}")
