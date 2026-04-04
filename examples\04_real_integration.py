"""
Example 4: Real LangGraph + AutoGen integration (runnable if you have API keys)

This example shows exactly how to wire real LangGraph and AutoGen agents
together using AgentLink, with actual LLM calls.

Requirements:
    pip install langchain-core langgraph openai pyautogen

Set environment variable:
    OPENAI_API_KEY=your-key

Run: python examples/04_real_integration.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlink import AgentBus
from agentlink.adapters import LangGraphAdapter, AutoGenAdapter, GenericAdapter


def build_langgraph_planner():
    """Build a real LangGraph planning agent."""
    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
        from langgraph.graph import StateGraph, MessagesState, START, END

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        def plan_node(state: MessagesState):
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("planner", plan_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", END)
        compiled = graph.compile()

        return LangGraphAdapter(
            graph=compiled,
            agent_id="lg-planner",
            capabilities=["planning"],
            description="LangGraph GPT-4o-mini planner",
        ).as_node()

    except ImportError as e:
        print(f"  [SKIP] LangGraph not installed: {e}")
        return None


def build_autogen_analyst():
    """Build a real AutoGen analysis agent."""
    try:
        import autogen

        llm_config = {
            "model": "gpt-4o-mini",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        }

        assistant = autogen.AssistantAgent(
            name="analyst",
            llm_config={"config_list": [llm_config]},
            system_message="You are a data analyst. Analyze inputs and provide clear insights.",
        )
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            code_execution_config=False,
        )

        return AutoGenAdapter(
            agent=assistant,
            agent_id="ag-analyst",
            initiator=user_proxy,
            capabilities=["data-analysis"],
            description="AutoGen GPT-4o-mini analyst",
            max_turns=3,
            silent=True,
        ).as_node()

    except ImportError as e:
        print(f"  [SKIP] AutoGen not installed: {e}")
        return None


def build_fallback_agent(name: str, role: str):
    """Fallback mock agent when frameworks aren't installed."""
    def handler(text: str) -> str:
        return f"[{role}] Mock response for: {text[:80]}"
    return GenericAdapter(fn=handler, agent_id=name, capabilities=[role.lower()]).as_node()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AgentLink — Real Integration Demo")
    print("="*60)

    bus = AgentBus("real-integration")

    # Try real frameworks, fall back to mocks
    planner = build_langgraph_planner() or build_fallback_agent("lg-planner", "planning")
    analyst = build_autogen_analyst()   or build_fallback_agent("ag-analyst", "analysis")

    # Always-available custom agent
    writer = GenericAdapter(
        fn=lambda text: f"[Custom Writer] Wrote article about: {text[:80]}",
        agent_id="writer",
        capabilities=["writing"],
    ).as_node()

    bus.register_many(planner, analyst, writer)
    bus.print_status()

    task = "Explain the significance of inter-agent communication protocols in 2026"

    print(f"\nTask: {task}\n")
    print("-" * 60)

    plan = planner.send("ag-analyst", task)
    print(f"Analyst: {plan.content}\n")

    report = analyst.send("writer", str(plan.content))
    print(f"Writer: {report.content}\n")
