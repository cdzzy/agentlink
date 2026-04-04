"""
AgentLink Protocol — Core message types and envelope format.

This is the heart of AgentLink: a framework-agnostic message format
that any agent, regardless of which framework it was built with,
can send and receive.

Design principles:
  - Self-describing: every message carries enough context to be routed
  - Extensible: metadata dict for framework-specific payloads
  - Minimal: only what's necessary, nothing more
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Message Types ───────────────────────────────────────────────────────────

class MessageType(str, Enum):
    """
    The fundamental message types in AgentLink.

    Modelled after common inter-process communication patterns,
    adapted for the agent domain.
    """
    # Core request/reply
    REQUEST  = "request"   # Ask another agent to do something
    REPLY    = "reply"     # Response to a REQUEST
    ERROR    = "error"     # Something went wrong

    # Streaming (for long-running tasks)
    STREAM_START  = "stream_start"   # Begin a streaming response
    STREAM_CHUNK  = "stream_chunk"   # A chunk of streamed content
    STREAM_END    = "stream_end"     # Streaming complete

    # Events (fire-and-forget, no reply expected)
    EVENT    = "event"     # Notify interested agents of something that happened

    # Discovery & Handshake
    PING         = "ping"          # Are you alive?
    PONG         = "pong"          # Yes, I'm alive
    ANNOUNCE     = "announce"      # I exist and have these capabilities
    WITHDRAW     = "withdraw"      # I'm going offline

    # Delegation
    DELEGATE = "delegate"   # Transfer responsibility for a task
    FORWARD  = "forward"    # Route a message to a different agent


# ── Agent Addressing ─────────────────────────────────────────────────────────

@dataclass
class AgentAddress:
    """
    Identifies an agent on the AgentLink network.

    Format: agent_id@namespace/capability
    Example: "researcher@my-app/web-search"
             "summarizer@production"
             "planner@*"   (* = any namespace)
    """
    agent_id: str
    namespace: str = "default"
    capability: Optional[str] = None   # optional capability filter for routing

    def __str__(self) -> str:
        s = f"{self.agent_id}@{self.namespace}"
        if self.capability:
            s += f"/{self.capability}"
        return s

    @classmethod
    def parse(cls, address: str) -> "AgentAddress":
        """Parse an address string into an AgentAddress."""
        capability = None
        if "/" in address:
            address, capability = address.rsplit("/", 1)
        if "@" in address:
            agent_id, namespace = address.split("@", 1)
        else:
            agent_id, namespace = address, "default"
        return cls(agent_id=agent_id, namespace=namespace, capability=capability)

    @classmethod
    def local(cls, agent_id: str) -> "AgentAddress":
        return cls(agent_id=agent_id, namespace="default")

    def matches(self, other: "AgentAddress") -> bool:
        """Check if this address matches another (supports wildcards)."""
        if self.agent_id != "*" and self.agent_id != other.agent_id:
            return False
        if self.namespace != "*" and self.namespace != other.namespace:
            return False
        if self.capability and self.capability != other.capability:
            return False
        return True


# ── Core Message ──────────────────────────────────────────────────────────────

@dataclass
class AgentMessage:
    """
    The fundamental unit of communication between agents.

    Every inter-agent communication in AgentLink is an AgentMessage.
    Think of it as the HTTP Request/Response of the agent world.

    Example:
        msg = AgentMessage(
            type=MessageType.REQUEST,
            sender=AgentAddress("planner", "my-app"),
            recipient=AgentAddress("researcher", "my-app"),
            content="Search for the latest AI news and summarize it.",
        )
    """
    type: MessageType
    sender: AgentAddress
    recipient: AgentAddress
    content: Any                          # The actual payload (str, dict, or anything)

    # Routing & correlation
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None  # Links reply to original request
    parent_id: Optional[str] = None       # For delegation chains

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl: Optional[int] = None             # Time-to-live in seconds (None = no limit)

    # Content typing
    content_type: str = "text/plain"      # MIME-like: "text/plain", "application/json", "agent/task"

    def reply(self, content: Any, msg_type: MessageType = MessageType.REPLY) -> "AgentMessage":
        """Create a reply to this message."""
        return AgentMessage(
            type=msg_type,
            sender=self.recipient,
            recipient=self.sender,
            content=content,
            correlation_id=self.id,
            parent_id=self.id,
            content_type="text/plain" if isinstance(content, str) else "application/json",
        )

    def error(self, error_message: str, code: str = "AGENT_ERROR") -> "AgentMessage":
        """Create an error reply to this message."""
        return AgentMessage(
            type=MessageType.ERROR,
            sender=self.recipient,
            recipient=self.sender,
            content={"error": error_message, "code": code, "original_id": self.id},
            correlation_id=self.id,
            content_type="application/json",
        )

    def forward_to(self, new_recipient: AgentAddress) -> "AgentMessage":
        """Create a forwarded copy of this message."""
        return AgentMessage(
            type=MessageType.FORWARD,
            sender=self.sender,
            recipient=new_recipient,
            content=self.content,
            parent_id=self.id,
            correlation_id=self.correlation_id,
            content_type=self.content_type,
            metadata={**self.metadata, "_forwarded_from": str(self.recipient)},
        )

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": str(self.sender),
            "recipient": str(self.recipient),
            "content": self.content,
            "correlation_id": self.correlation_id,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp.isoformat(),
            "content_type": self.content_type,
            "metadata": self.metadata,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            sender=AgentAddress.parse(data["sender"]),
            recipient=AgentAddress.parse(data["recipient"]),
            content=data["content"],
            correlation_id=data.get("correlation_id"),
            parent_id=data.get("parent_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            content_type=data.get("content_type", "text/plain"),
            metadata=data.get("metadata", {}),
            ttl=data.get("ttl"),
        )

    def __repr__(self) -> str:
        content_preview = str(self.content)[:60] + "..." if len(str(self.content)) > 60 else str(self.content)
        return (
            f"AgentMessage(type={self.type.value!r}, "
            f"from={self.sender}, to={self.recipient}, "
            f"content={content_preview!r})"
        )


# ── Message Envelope ──────────────────────────────────────────────────────────

@dataclass
class MessageEnvelope:
    """
    A signed, versioned wrapper around an AgentMessage for transport.

    The envelope handles:
    - Protocol versioning
    - Hop counting (prevent infinite routing loops)
    - Route tracing (for debugging)
    - Optional signing (for trust verification)
    """
    message: AgentMessage
    protocol_version: str = "agentlink/1.0"
    hop_count: int = 0
    max_hops: int = 10
    route: List[str] = field(default_factory=list)   # addresses visited
    signature: Optional[str] = None                   # for future auth support

    def record_hop(self, node_id: str) -> "MessageEnvelope":
        """Record that this envelope passed through a node."""
        self.hop_count += 1
        self.route.append(node_id)
        return self

    def is_loop_detected(self) -> bool:
        """Detect if a message is routing in circles."""
        return self.hop_count >= self.max_hops

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "hop_count": self.hop_count,
            "max_hops": self.max_hops,
            "route": self.route,
            "signature": self.signature,
            "message": self.message.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageEnvelope":
        return cls(
            message=AgentMessage.from_dict(data["message"]),
            protocol_version=data.get("protocol_version", "agentlink/1.0"),
            hop_count=data.get("hop_count", 0),
            max_hops=data.get("max_hops", 10),
            route=data.get("route", []),
            signature=data.get("signature"),
        )

    def wrap(self, message: AgentMessage) -> "MessageEnvelope":
        return MessageEnvelope(message=message)
