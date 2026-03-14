"""
CrewAI Adapter — wrap a CrewAI Crew for AgentLink.

Translates AgentLink messages into CrewAI task inputs and
extracts the final crew output.

Usage:
    from crewai import Crew, Agent, Task
    from agentlink.adapters import CrewAIAdapter

    researcher = Agent(role="Researcher", goal="Find information", ...)
    writer = Agent(role="Writer", goal="Write reports", ...)
    crew = Crew(agents=[researcher, writer], tasks=[...])

    adapter = CrewAIAdapter(
        crew=crew,
        agent_id="research-crew",
        capabilities=["research", "report-writing"],
    )
    node = adapter.as_node()
    bus.register(node)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from agentlink.adapters.base import BaseAdapter
from agentlink.protocol.capability import AgentCapability
from agentlink.protocol.message import AgentMessage


class CrewAIAdapter(BaseAdapter):
    """
    Adapter for CrewAI Crew objects.

    Handles dynamic task creation from incoming messages,
    since CrewAI tasks are typically defined at build time
    but we need to inject runtime inputs.
    """

    def __init__(
        self,
        crew: Any,                                      # CrewAI Crew
        agent_id: str,
        namespace: str = "default",
        capabilities: Optional[List[Union[str, AgentCapability]]] = None,
        description: str = "",
        task_input_key: str = "topic",                  # input variable name in task descriptions
        inputs_builder: Optional[Callable[[AgentMessage], Dict]] = None,
        result_extractor: Optional[Callable[[Any], str]] = None,
    ):
        """
        Args:
            crew: A CrewAI Crew instance.
            agent_id: Unique ID for this agent on the bus.
            namespace: Logical group.
            capabilities: Capability declarations.
            description: Human description.
            task_input_key: The variable name in task description templates
                            to inject the message content into.
                            e.g., if description="Research {topic}", key is "topic".
            inputs_builder: Custom function to build the inputs dict from a message.
            result_extractor: Custom function to extract the final answer.
        """
        super().__init__(agent_id, namespace, capabilities, description)
        self._crew = crew
        self._task_input_key = task_input_key
        self._inputs_builder = inputs_builder or self._default_inputs_builder
        self._result_extractor = result_extractor or self._default_extractor

    def _invoke(self, message: AgentMessage) -> str:
        inputs = self._inputs_builder(message)
        result = self._crew.kickoff(inputs=inputs)
        return self._result_extractor(result)

    def _default_inputs_builder(self, message: AgentMessage) -> Dict[str, Any]:
        """Build a simple inputs dict from the message content."""
        content = str(message.content)
        return {
            self._task_input_key: content,
            "agentlink_sender": str(message.sender),
            "agentlink_message_id": message.id,
        }

    def _default_extractor(self, result: Any) -> str:
        """Extract the final output from CrewAI's kickoff result."""
        try:
            # CrewAI v0.30+ returns a CrewOutput object
            if hasattr(result, "raw"):
                return result.raw
            if hasattr(result, "final_output"):
                return result.final_output
            return str(result)
        except Exception:
            return str(result)

    @classmethod
    def from_crew(
        cls,
        crew: Any,
        agent_id: str,
        capabilities: Optional[List[str]] = None,
        namespace: str = "default",
        **kwargs,
    ) -> "CrewAIAdapter":
        """Convenience constructor."""
        return cls(
            crew=crew,
            agent_id=agent_id,
            namespace=namespace,
            capabilities=capabilities,
            **kwargs,
        )
