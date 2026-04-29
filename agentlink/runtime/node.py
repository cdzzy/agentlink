"""
AgentNode — wraps any agent function into an AgentLink-compatible node.

This is the adapter layer between "your agent code" and "the protocol".
Regardless of whether your agent is LangGraph, AutoGen, CrewAI, or
plain Python, wrapping it in an AgentNode gives it a network identity
and the ability to speak AgentLink.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

from agentlink.protocol.capability import AgentCapability, CapabilitySet
from agentlink.protocol.message import AgentAddress, AgentMessage, MessageType

logger = logging.getLogger(__name__)


class AgentNode:
    """
    An AgentLink network node wrapping any agent implementation.

    Usage:
        # Wrap a simple function agent
        node = AgentNode(
            agent_id="researcher",
            handler=my_research_fn,
            capabilities=["web-search", "summarize"],
        )

        # Wrap a LangGraph agent
        node = AgentNode(
            agent_id="planner",
            handler=langgraph_wrapper,
            namespace="production",
            capabilities=["planning"],
        )

        # Register and start communicating
        bus.register(node)
        reply = node.send("researcher", "Find the latest AI news")
    """

    def __init__(
        self,
        agent_id: str,
        handler: Callable[[AgentMessage], Any],
        namespace: str = "default",
        capabilities: Optional[List[Union[str, AgentCapability]]] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            agent_id: Unique identifier for this agent.
            handler: The callable that processes incoming messages.
                     Receives an AgentMessage, returns a string or dict.
            namespace: Logical grouping (e.g., "production", "research-team").
            capabilities: What this agent can do.
            description: Human-readable description.
            metadata: Arbitrary metadata dict.
        """
        self.agent_id = agent_id
        self.namespace = namespace
        self.description = description
        self.metadata = metadata or {}
        self.address = AgentAddress(agent_id, namespace)

        # Build capability set
        self.capabilities = CapabilitySet()
        for cap in (capabilities or []):
            if isinstance(cap, str):
                self.capabilities.add(AgentCapability(name=cap))
            else:
                self.capabilities.add(cap)

        # The actual agent logic
        self._handler = handler

        # Bus reference (set when registered)
        self._bus: Optional[Any] = None

        # Message inbox & stats
        self._inbox: List[AgentMessage] = []
        self._inbox_lock = threading.Lock()
        self._stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "errors": 0,
        }

    # ── Internal: called by the bus ──────────────────────────────────────

    def _attach_bus(self, bus: Any):
        self._bus = bus

    def _receive(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Called by the bus to deliver a message to this node."""
        self._stats["messages_received"] += 1

        with self._inbox_lock:
            self._inbox.append(message)

        # Skip processing for PING — auto-reply with PONG
        if message.type == MessageType.PING:
            return message.reply("pong", MessageType.PONG)

        # Skip processing for events (fire-and-forget)
        if message.type == MessageType.EVENT:
            try:
                self._handler(message)
            except Exception as e:
                logger.warning(f"[{self.agent_id}] Error handling event: {e}")
            return None

        # Process REQUEST and DELEGATE
        try:
            result = self._handler(message)
            reply = self._build_reply(message, result)
            return reply
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"[{self.agent_id}] Handler error: {e}")
            return message.error(str(e))

    def _build_reply(self, original: AgentMessage, result: Any) -> AgentMessage:
        """Convert handler return value into a proper reply message."""
        if isinstance(result, AgentMessage):
            return result
        if isinstance(result, dict):
            content = result.get("output", result.get("response", result))
            content_type = "application/json" if isinstance(content, dict) else "text/plain"
        else:
            content = str(result) if result is not None else ""
            content_type = "text/plain"

        return AgentMessage(
            type=MessageType.REPLY,
            sender=self.address,
            recipient=original.sender,
            content=content,
            correlation_id=original.id,
            content_type=content_type,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def send(
        self,
        recipient: Union[str, AgentAddress],
        content: Any,
        msg_type: MessageType = MessageType.REQUEST,
        timeout: float = 30.0,
        **kwargs,
    ) -> Optional[AgentMessage]:
        """
        Send a message to another agent and wait for a reply.

        Args:
            recipient: Agent ID string or AgentAddress.
            content: Message payload (string or dict).
            msg_type: Type of message to send.
            timeout: Seconds to wait for a reply.
            **kwargs: Extra fields for AgentMessage.

        Returns:
            The reply AgentMessage, or None if no reply expected.

        Example:
            reply = node.send("researcher", "Find AI news from this week")
            print(reply.content)
        """
        if self._bus is None:
            raise RuntimeError(
                f"Agent '{self.agent_id}' is not connected to a bus. "
                "Call bus.register(node) first."
            )

        if isinstance(recipient, str):
            # Support "agent_id@namespace" or just "agent_id"
            recipient = AgentAddress.parse(recipient) if "@" in recipient else AgentAddress(recipient, self.namespace)

        msg = AgentMessage(
            type=msg_type,
            sender=self.address,
            recipient=recipient,
            content=content,
            content_type="text/plain" if isinstance(content, str) else "application/json",
            **kwargs,
        )

        self._stats["messages_sent"] += 1
        return self._bus._route(msg, sender_node=self, timeout=timeout)

    def send_event(self, recipient: Union[str, AgentAddress], event_type: str, data: Any = None):
        """Fire-and-forget: broadcast an event to another agent."""
        content = {"event_type": event_type, "data": data}
        self.send(recipient, content, msg_type=MessageType.EVENT, timeout=0)

    def broadcast(self, content: Any, event_type: str = "broadcast"):
        """Broadcast a message to all agents in the same namespace."""
        if self._bus is None:
            raise RuntimeError("Not connected to a bus.")
        self._bus._broadcast(self, content, event_type)

    def ping(self, recipient: Union[str, AgentAddress], timeout: float = 5.0) -> bool:
        """Check if another agent is alive. Returns True if PONG received."""
        try:
            reply = self.send(recipient, "ping", msg_type=MessageType.PING, timeout=timeout)
            return reply is not None and reply.type == MessageType.PONG
        except Exception:
            return False

    @property
    def stats(self) -> Dict[str, int]:
        return self._stats.copy()

    @property
    def inbox(self) -> List[AgentMessage]:
        with self._inbox_lock:
            return self._inbox.copy()

    def info(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "address": str(self.address),
            "description": self.description,
            "capabilities": self.capabilities.names(),
            "stats": self.stats,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        caps = self.capabilities.names()
        return f"AgentNode(id={self.agent_id!r}, ns={self.namespace!r}, caps={caps})"
