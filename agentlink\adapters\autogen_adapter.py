"""
AutoGen Adapter — wrap a Microsoft AutoGen agent/team for AgentLink.

Translates AgentLink messages into AutoGen's task-based API
and extracts the final response.

Usage:
    import autogen
    from agentlink.adapters import AutoGenAdapter

    assistant = autogen.AssistantAgent("assistant", llm_config=llm_config)
    user_proxy = autogen.UserProxyAgent("user_proxy", ...)

    adapter = AutoGenAdapter(
        agent=assistant,
        initiator=user_proxy,
        agent_id="code-reviewer",
        capabilities=["code-review", "code-generation"],
    )
    node = adapter.as_node()
    bus.register(node)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from agentlink.adapters.base import BaseAdapter
from agentlink.protocol.capability import AgentCapability
from agentlink.protocol.message import AgentMessage


class AutoGenAdapter(BaseAdapter):
    """
    Adapter for Microsoft AutoGen agents.

    Supports both single AssistantAgent and multi-agent GroupChat setups.
    """

    def __init__(
        self,
        agent: Any,                              # AutoGen AssistantAgent or GroupChatManager
        agent_id: str,
        namespace: str = "default",
        capabilities: Optional[List[Union[str, AgentCapability]]] = None,
        description: str = "",
        initiator: Optional[Any] = None,         # UserProxyAgent for initiating chats
        max_turns: int = 10,
        silent: bool = True,                     # suppress AutoGen console output
        result_extractor: Optional[Callable] = None,
    ):
        """
        Args:
            agent: AutoGen agent (AssistantAgent or GroupChatManager).
            agent_id: Unique ID for this agent on the bus.
            namespace: Logical group.
            capabilities: Capability declarations.
            description: Human description.
            initiator: UserProxyAgent used to kick off conversations.
            max_turns: Max AutoGen conversation turns per invocation.
            silent: Whether to suppress AutoGen's verbose output.
            result_extractor: Custom function to extract the final answer
                              from AutoGen's chat history.
        """
        super().__init__(agent_id, namespace, capabilities, description)
        self._agent = agent
        self._initiator = initiator
        self._max_turns = max_turns
        self._silent = silent
        self._result_extractor = result_extractor or self._default_extractor

    def _invoke(self, message: AgentMessage) -> str:
        content = str(message.content)

        if self._initiator is not None:
            # Multi-agent conversation: initiator kicks things off
            chat_result = self._initiator.initiate_chat(
                self._agent,
                message=content,
                max_turns=self._max_turns,
                silent=self._silent,
            )
            return self._result_extractor(chat_result)
        else:
            # Single-agent: call generate_reply directly
            try:
                messages = [{"role": "user", "content": content}]
                reply = self._agent.generate_reply(messages=messages)
                return reply if isinstance(reply, str) else str(reply)
            except Exception as e:
                raise RuntimeError(f"AutoGen agent failed: {e}") from e

    def _default_extractor(self, chat_result: Any) -> str:
        """Extract the last assistant message from AutoGen chat result."""
        try:
            # AutoGen v0.4+ ChatResult
            if hasattr(chat_result, "summary"):
                return chat_result.summary or ""
            # chat_history list
            if hasattr(chat_result, "chat_history") and chat_result.chat_history:
                for msg in reversed(chat_result.chat_history):
                    role = msg.get("role", "")
                    if role in ("assistant", "user"):
                        content = msg.get("content", "")
                        if content and content.strip():
                            return content
            return str(chat_result)
        except Exception:
            return str(chat_result)

    @classmethod
    def from_assistant(
        cls,
        assistant_agent: Any,
        agent_id: str,
        initiator: Any,
        capabilities: Optional[List[str]] = None,
        namespace: str = "default",
        **kwargs,
    ) -> "AutoGenAdapter":
        """Convenience constructor for a single AssistantAgent setup."""
        return cls(
            agent=assistant_agent,
            agent_id=agent_id,
            namespace=namespace,
            capabilities=capabilities,
            initiator=initiator,
            **kwargs,
        )
