"""
Base adapter — the interface all framework adapters must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from agentlink.protocol.capability import AgentCapability
from agentlink.protocol.message import AgentMessage
from agentlink.runtime.node import AgentNode


class BaseAdapter(ABC):
    """
    Abstract base for all AgentLink framework adapters.

    An adapter's job:
    1. Take a framework-specific agent object
    2. Translate AgentLink messages into the framework's input format
    3. Translate the framework's output back into an AgentLink reply

    Subclass this to support any new framework.
    """

    def __init__(
        self,
        agent_id: str,
        namespace: str = "default",
        capabilities: Optional[List[Union[str, AgentCapability]]] = None,
        description: str = "",
    ):
        self.agent_id = agent_id
        self.namespace = namespace
        self.capabilities = capabilities or []
        self.description = description

    @abstractmethod
    def _invoke(self, message: AgentMessage) -> Any:
        """
        Invoke the underlying framework agent with the given message.
        Return the raw framework output.
        """
        ...

    def _normalize_output(self, raw: Any) -> str:
        """Convert framework output to a string. Override as needed."""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            return raw.get("output", raw.get("response", raw.get("content", str(raw))))
        return str(raw) if raw is not None else ""

    def as_node(self) -> AgentNode:
        """
        Convert this adapter into an AgentLink AgentNode.
        The node can then be registered on any AgentBus.
        """
        def handler(message: AgentMessage) -> Any:
            return self._invoke(message)

        return AgentNode(
            agent_id=self.agent_id,
            handler=handler,
            namespace=self.namespace,
            capabilities=self.capabilities,
            description=self.description,
            metadata={"adapter": type(self).__name__},
        )
