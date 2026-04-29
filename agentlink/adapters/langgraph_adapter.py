"""
LangGraph Adapter — wrap a LangGraph StateGraph agent for AgentLink.

Translates AgentLink messages into LangGraph's HumanMessage input format
and extracts the response from the graph's output state.

Usage:
    from langgraph.graph import StateGraph
    from agentlink.adapters import LangGraphAdapter

    graph = StateGraph(...)
    # ... build your graph ...
    compiled = graph.compile()

    adapter = LangGraphAdapter(
        graph=compiled,
        agent_id="planner",
        capabilities=["planning", "task-decomposition"],
    )
    node = adapter.as_node()
    bus.register(node)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from agentlink.adapters.base import BaseAdapter
from agentlink.protocol.capability import AgentCapability
from agentlink.protocol.message import AgentMessage


class LangGraphAdapter(BaseAdapter):
    """
    Adapter for LangGraph compiled graphs.

    Works with any LangGraph graph that follows the standard
    messages-in / messages-out state pattern.
    """

    def __init__(
        self,
        graph: Any,                              # CompiledGraph from LangGraph
        agent_id: str,
        namespace: str = "default",
        capabilities: Optional[List[Union[str, AgentCapability]]] = None,
        description: str = "",
        input_key: str = "messages",             # state key for input
        output_key: str = "messages",            # state key for output
        config: Optional[Dict[str, Any]] = None, # LangGraph run config
    ):
        """
        Args:
            graph: A compiled LangGraph graph (graph.compile() result).
            agent_id: Unique ID for this agent on the bus.
            namespace: Logical group (e.g. "research-team").
            capabilities: What this agent can do.
            description: Human description.
            input_key: The state key to put input into (usually "messages").
            output_key: The state key to read output from (usually "messages").
            config: Optional LangGraph run configuration dict.
        """
        super().__init__(agent_id, namespace, capabilities, description)
        self._graph = graph
        self._input_key = input_key
        self._output_key = output_key
        self._config = config or {}

    def _invoke(self, message: AgentMessage) -> str:
        """
        Translate AgentLink message → LangGraph invocation → string output.

        Handles both LangChain message objects (if installed) and
        plain dict fallback.
        """
        content = message.content
        if isinstance(content, dict):
            content = content.get("content", str(content))
        else:
            content = str(content)

        # Build input state
        try:
            # Try using LangChain message types if available
            from langchain_core.messages import HumanMessage
            input_state = {self._input_key: [HumanMessage(content=content)]}
        except ImportError:
            # Fallback: plain dict format
            input_state = {self._input_key: [{"role": "user", "content": content}]}

        # Inject AgentLink metadata into the run
        run_config = {
            **self._config,
            "metadata": {
                **(self._config.get("metadata", {})),
                "agentlink_message_id": message.id,
                "agentlink_sender": str(message.sender),
            },
        }

        result = self._graph.invoke(input_state, config=run_config)

        # Extract output
        output_messages = result.get(self._output_key, [])
        if output_messages:
            last = output_messages[-1]
            if hasattr(last, "content"):
                return last.content
            if isinstance(last, dict):
                return last.get("content", str(last))
        return str(result)

    @classmethod
    def from_graph(
        cls,
        graph: Any,
        agent_id: str,
        capabilities: Optional[List[str]] = None,
        namespace: str = "default",
        **kwargs,
    ) -> "LangGraphAdapter":
        """Convenience constructor."""
        return cls(
            graph=graph,
            agent_id=agent_id,
            namespace=namespace,
            capabilities=capabilities,
            **kwargs,
        )
