"""
AgentBus — the message routing backbone of AgentLink.

The bus is the central switchboard. Agents register here,
and the bus routes messages between them based on address,
capability, or namespace.

Think of it as:
  - A post office (reliable delivery)
  - A telephone exchange (routing)
  - A service mesh (discovery + load balancing)

All in one, lightweight, zero-dependency.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Union

from agentlink.protocol.message import AgentAddress, AgentMessage, MessageType
from agentlink.protocol.routing import RoutingStrategy
from agentlink.runtime.registry import AgentRegistry

logger = logging.getLogger(__name__)


class DeliveryError(Exception):
    """Raised when a message cannot be delivered."""
    pass


class TimeoutError(Exception):
    """Raised when waiting for a reply times out."""
    pass


class AgentBus:
    """
    The AgentLink message bus.

    Handles:
    - Agent registration and deregistration
    - Message routing (direct, capability-based, broadcast)
    - Reply correlation (request-reply pattern)
    - Middleware pipeline (logging, auth, tracing, etc.)

    Usage:
        bus = AgentBus()

        # Register agents
        bus.register(planner_node)
        bus.register(researcher_node)
        bus.register(writer_node)

        # Now agents can talk to each other via node.send(...)

        # Inspect the bus
        bus.print_status()
    """

    def __init__(self, name: str = "default-bus"):
        self.name = name
        self.registry = AgentRegistry()

        # Pending reply waiters: correlation_id -> (Event, result_holder)
        self._pending: Dict[str, Dict] = {}
        self._pending_lock = threading.Lock()

        # Middleware chain
        self._middleware: List[Callable] = []

        # Message log (deque with auto-truncate at 1000)
        self._message_log: Deque[Dict] = deque(maxlen=1000)
        self._log_lock = threading.Lock()

        # Stats
        self._stats = {
            "messages_routed": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "broadcasts": 0,
        }

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, node: Any) -> "AgentBus":
        """
        Register an AgentNode on this bus.

        After registration, the node can send and receive messages.

        Example:
            bus.register(planner_node)
            bus.register(researcher_node)
        """
        node._attach_bus(self)
        self.registry.register(node)
        logger.info(f"[bus:{self.name}] Registered: {node.agent_id}@{node.namespace} "
                    f"caps={node.capabilities.names()}")

        # Announce to the namespace
        self._announce(node)
        return self

    def deregister(self, node: Any) -> "AgentBus":
        """Remove an agent from the bus."""
        self.registry.deregister(node.agent_id, node.namespace)
        logger.info(f"[bus:{self.name}] Deregistered: {node.agent_id}@{node.namespace}")
        return self

    def register_many(self, *nodes) -> "AgentBus":
        for node in nodes:
            self.register(node)
        return self

    # ── Routing ───────────────────────────────────────────────────────────

    def _route(
        self,
        message: AgentMessage,
        sender_node: Any = None,
        timeout: float = 30.0,
    ) -> Optional[AgentMessage]:
        """
        Main routing method. Finds the target node and delivers the message.
        For REQUEST/DELEGATE messages, waits for a reply.
        """
        self._stats["messages_routed"] += 1
        self._log_message(message, "routed")

        # Run middleware
        for mw in self._middleware:
            message = mw(message)
            if message is None:
                return None   # middleware blocked the message

        # Resolve recipient
        target_node = self._resolve_recipient(message.recipient, message)
        if target_node is None:
            self._stats["messages_failed"] += 1
            raise DeliveryError(
                f"No agent found for address: {message.recipient}\n"
                f"Registered agents: {[r.address_str for r in self.registry.all_agents()]}"
            )

        # Fire-and-forget for events and no-reply types
        if message.type in (MessageType.EVENT, MessageType.ANNOUNCE, MessageType.WITHDRAW):
            target_node._receive(message)
            self._stats["messages_delivered"] += 1
            return None

        # For PING/REQUEST/DELEGATE: deliver and get reply
        if timeout <= 0:
            target_node._receive(message)
            return None

        reply = target_node._receive(message)
        self._stats["messages_delivered"] += 1

        if reply:
            self._log_message(reply, "reply")

        return reply

    def _resolve_recipient(self, address: AgentAddress, message: AgentMessage) -> Optional[Any]:
        """Resolve an AgentAddress to an actual AgentNode."""
        # Direct lookup first
        node = self.registry.find(address.agent_id, address.namespace)
        if node:
            return node

        # Capability-based routing: if agent_id looks like a capability
        capable = self.registry.find_by_capability(address.agent_id, address.namespace)
        if capable:
            return capable[0]   # first match (could do round-robin here)

        # Try without namespace constraint
        node = self.registry.find(address.agent_id, "default")
        if node:
            return node

        return None

    def _broadcast(self, sender_node: Any, content: Any, event_type: str = "broadcast"):
        """Broadcast a message to all agents in the sender's namespace."""
        self._stats["broadcasts"] += 1
        targets = self.registry.find_by_namespace(sender_node.namespace)
        for target in targets:
            if target.agent_id == sender_node.agent_id:
                continue  # don't send to self
            msg = AgentMessage(
                type=MessageType.EVENT,
                sender=sender_node.address,
                recipient=target.address,
                content={"event_type": event_type, "data": content},
                content_type="application/json",
            )
            try:
                target._receive(msg)
            except Exception as e:
                logger.warning(f"[bus] Broadcast delivery failed to {target.agent_id}: {e}")

    def _announce(self, node: Any):
        """Announce a new agent to all agents in its namespace."""
        targets = self.registry.find_by_namespace(node.namespace)
        for target in targets:
            if target.agent_id == node.agent_id:
                continue
            msg = AgentMessage(
                type=MessageType.ANNOUNCE,
                sender=node.address,
                recipient=target.address,
                content={
                    "agent_id": node.agent_id,
                    "namespace": node.namespace,
                    "capabilities": node.capabilities.names(),
                    "description": node.description,
                },
                content_type="application/json",
            )
            try:
                target._receive(msg)
            except Exception:
                pass

    # ── Direct bus-level send ─────────────────────────────────────────────

    def send(
        self,
        sender_id: str,
        recipient_id: str,
        content: Any,
        namespace: str = "default",
        timeout: float = 30.0,
    ) -> Optional[AgentMessage]:
        """
        Send a message directly through the bus (without a sender node).

        Useful for testing or external systems injecting messages.

        Example:
            reply = bus.send("orchestrator", "researcher", "Find AI news")
        """
        sender_addr   = AgentAddress(sender_id, namespace)
        recipient_addr = AgentAddress(recipient_id, namespace)
        msg = AgentMessage(
            type=MessageType.REQUEST,
            sender=sender_addr,
            recipient=recipient_addr,
            content=content,
        )
        return self._route(msg, timeout=timeout)

    # ── Middleware ────────────────────────────────────────────────────────

    def use(self, middleware: Callable[[AgentMessage], Optional[AgentMessage]]) -> "AgentBus":
        """
        Add a middleware function to the routing pipeline.

        Middleware receives a message, can modify it, and must return it
        (or return None to drop the message).

        Example:
            def logging_middleware(msg):
                print(f"[ROUTE] {msg.sender} -> {msg.recipient}")
                return msg   # must return msg to continue

            bus.use(logging_middleware)
        """
        self._middleware.append(middleware)
        return self

    # ── Logging & Introspection ───────────────────────────────────────────

    def _log_message(self, message: AgentMessage, event: str):
        with self._log_lock:
            entry = {
                "event": event,
                "id": message.id,
                "type": message.type.value,
                "from": str(message.sender),
                "to": str(message.recipient),
                "at": time.time(),
            }
            self._message_log.append(entry)
            # deque with maxlen=1000 auto-truncates when full

    @property
    def message_log(self) -> List[Dict]:
        with self._log_lock:
            return self._message_log.copy()

    @property
    def stats(self) -> Dict:
        return {**self._stats, "registered_agents": len(self.registry)}

    def print_status(self):
        """Print a human-readable status of the bus."""
        print(f"\n{'='*55}")
        print(f"  AgentLink Bus: {self.name!r}")
        print(f"{'='*55}")
        summary = self.registry.summary()
        print(f"  Registered agents : {summary['total_agents']}")
        print(f"  Namespaces        : {list(summary['namespaces'].keys())}")
        print(f"  Messages routed   : {self._stats['messages_routed']}")
        print(f"  Broadcasts        : {self._stats['broadcasts']}")
        print(f"\n  Agents:")
        for record in self.registry.all_agents():
            caps = ", ".join(record.capabilities) if record.capabilities else "(none)"
            print(f"    - {record.address_str:<30}  caps: {caps}")
        print(f"{'='*55}\n")

    def __repr__(self) -> str:
        return f"AgentBus(name={self.name!r}, agents={len(self.registry)})"

