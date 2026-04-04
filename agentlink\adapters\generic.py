"""
Generic Adapter — wrap any callable as an AgentLink node.

The simplest adapter: just give it any Python callable
and it becomes an AgentLink agent.

Usage:
    def my_agent(input_text: str) -> str:
        return f"Processed: {input_text}"

    adapter = GenericAdapter(
        fn=my_agent,
        agent_id="my-agent",
        capabilities=["general"],
    )
    node = adapter.as_node()
    bus.register(node)
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

from agentlink.adapters.base import BaseAdapter
from agentlink.protocol.capability import AgentCapability
from agentlink.protocol.message import AgentMessage


class GenericAdapter(BaseAdapter):
    """
    Wraps any Python callable as an AgentLink node.

    Your function can accept:
    - (input_text: str) -> str | dict
    - (message: AgentMessage) -> str | dict    (if accept_message=True)
    """

    def __init__(
        self,
        fn: Callable,
        agent_id: str,
        namespace: str = "default",
        capabilities: Optional[List[Union[str, AgentCapability]]] = None,
        description: str = "",
        accept_message: bool = False,   # pass full AgentMessage instead of just content string
    ):
        super().__init__(agent_id, namespace, capabilities, description)
        self._fn = fn
        self._accept_message = accept_message

    def _invoke(self, message: AgentMessage) -> Any:
        if self._accept_message:
            return self._fn(message)
        else:
            content = str(message.content) if not isinstance(message.content, str) else message.content
            return self._fn(content)
