"""
A2A v1.0 Agent Card — generation and parsing (AgentLink A2A compatibility layer).

Implements the Agent Card discovery document of the A2A (Agent2Agent) protocol
v1.0: https://a2a-protocol.org/v1.0.0/specification/

Key v1.0 conventions implemented here:
  - The card is published at ``/.well-known/agent-card.json``
    (``AGENT_CARD_WELLKNOWN_PATH``).
  - There is no top-level ``url`` field: the primary service endpoint lives in
    ``supportedInterfaces[0].url`` (each interface carries its own
    ``protocolBinding`` and ``protocolVersion``). For convenience this module
    exposes a ``url`` property that reads/writes ``supportedInterfaces[0].url``.
  - ``capabilities`` is an object: ``streaming``, ``pushNotifications``,
    ``extendedAgentCard`` and ``extensions``.
  - ``signatures`` carries detached-JWS (RFC 7515) signatures over the
    RFC 8785-canonicalized card (see ``agentlink.a2a.verify``).

Parsing is tolerant: legacy v0.2/v0.3 cards that carry a top-level ``url`` /
``preferredTransport`` / ``protocolVersion`` are normalized into the v1.0
``supportedInterfaces`` model, and unknown top-level fields are preserved in
``extra`` for lossless round-trips.

Zero external dependencies — pure standard library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Well-known discovery path per the A2A v1.0 specification.
AGENT_CARD_WELLKNOWN_PATH = "/.well-known/agent-card.json"

# Protocol version emitted by default on generated interfaces.
A2A_PROTOCOL_VERSION = "1.0"

# Agent's own implementation version default (A2A requires a ``version`` field;
# this is the agent's version, not the protocol version).
DEFAULT_CARD_VERSION = "0.0.0"


def agent_card_url(base_url: str) -> str:
    """Return the well-known Agent Card URL for a server base URL."""
    return base_url.rstrip("/") + AGENT_CARD_WELLKNOWN_PATH


@dataclass
class AgentInterface:
    """
    One protocol endpoint of an agent (A2A v1.0 ``AgentInterface``).

    Attributes:
        url: Absolute service endpoint URL for this interface.
        protocolBinding: ``JSONRPC`` | ``GRPC`` | ``HTTP+JSON``.
        protocolVersion: A2A protocol version spoken on this interface.
        tenant: Optional default tenant for multi-tenant deployments.
    """

    url: str
    protocolBinding: str = "JSONRPC"  # noqa: N815 - A2A uses camelCase on the wire
    protocolVersion: str = A2A_PROTOCOL_VERSION  # noqa: N815
    tenant: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "url": self.url,
            "protocolBinding": self.protocolBinding,
            "protocolVersion": self.protocolVersion,
        }
        if self.tenant is not None:
            out["tenant"] = self.tenant
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentInterface":
        return cls(
            url=data.get("url", ""),
            protocolBinding=data.get("protocolBinding", "JSONRPC"),
            protocolVersion=data.get("protocolVersion", A2A_PROTOCOL_VERSION),
            tenant=data.get("tenant"),
        )


@dataclass
class AgentSkill:
    """A unit of capability an agent can perform (A2A ``AgentSkill``)."""

    id: str
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    inputModes: List[str] = field(default_factory=lambda: ["text"])  # noqa: N815
    outputModes: List[str] = field(default_factory=lambda: ["text"])  # noqa: N815

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.id, "name": self.name}
        if self.description:
            out["description"] = self.description
        if self.tags:
            out["tags"] = list(self.tags)
        if self.examples:
            out["examples"] = list(self.examples)
        out["inputModes"] = list(self.inputModes)
        out["outputModes"] = list(self.outputModes)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSkill":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            tags=list(data.get("tags", [])),
            examples=list(data.get("examples", [])),
            inputModes=list(data.get("inputModes", ["text"])),
            outputModes=list(data.get("outputModes", ["text"])),
        )


@dataclass
class AgentCapabilities:
    """Agent capability flags (A2A v1.0 ``AgentCapabilities``)."""

    streaming: bool = False
    pushNotifications: bool = False  # noqa: N815
    extendedAgentCard: bool = False  # noqa: N815
    extensions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "streaming": self.streaming,
            "pushNotifications": self.pushNotifications,
            "extendedAgentCard": self.extendedAgentCard,
        }
        if self.extensions:
            out["extensions"] = [dict(e) for e in self.extensions]
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCapabilities":
        return cls(
            streaming=bool(data.get("streaming", False)),
            pushNotifications=bool(data.get("pushNotifications", False)),
            extendedAgentCard=bool(data.get("extendedAgentCard", False)),
            extensions=[dict(e) for e in data.get("extensions", []) if isinstance(e, dict)],
        )


@dataclass
class AgentCardSignature:
    """
    A detached JWS signature over the card (A2A ``AgentCardSignature``).

    ``protected`` is the Base64url-encoded JOSE protected header and
    ``signature`` the Base64url-encoded JWS signature, both computed over the
    canonicalized card with the ``signatures`` field omitted.
    """

    protected: str
    signature: str
    header: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"protected": self.protected, "signature": self.signature}
        if self.header:
            out["header"] = dict(self.header)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCardSignature":
        return cls(
            protected=data.get("protected", ""),
            signature=data.get("signature", ""),
            header=data.get("header"),
        )


@dataclass
class AgentCard:
    """
    A2A v1.0 Agent Card — generation and parsing of the discovery document.

    Example:
        card = AgentCard(
            name="researcher",
            description="Research agent for market data",
            skills=[AgentSkill(id="web-search", name="Web Search")],
        )
        card.url = "https://agent.example.com/a2a"
        card.capabilities.streaming = True

        payload = card.to_json()                    # serve at /.well-known/agent-card.json
        parsed = AgentCard.from_json(payload)       # parse it back, losslessly
    """

    name: str
    description: str = ""
    version: str = DEFAULT_CARD_VERSION
    supportedInterfaces: List[AgentInterface] = field(default_factory=list)  # noqa: N815
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    skills: List[AgentSkill] = field(default_factory=list)
    provider: Optional[Dict[str, str]] = None
    documentationUrl: Optional[str] = None  # noqa: N815
    iconUrl: Optional[str] = None  # noqa: N815
    defaultInputModes: List[str] = field(default_factory=lambda: ["text"])  # noqa: N815
    defaultOutputModes: List[str] = field(default_factory=lambda: ["text"])  # noqa: N815
    securitySchemes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # noqa: N815
    security: List[Dict[str, List[str]]] = field(default_factory=list)
    signatures: List[AgentCardSignature] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)  # unknown fields, preserved verbatim

    # ── Primary endpoint convenience (v1.0: supportedInterfaces[0].url) ─────

    @property
    def url(self) -> str:
        """Primary service endpoint URL (``supportedInterfaces[0].url``)."""
        if self.supportedInterfaces:
            return self.supportedInterfaces[0].url
        # Legacy fallback: a top-level ``url`` kept in ``extra`` after parsing.
        return self.extra.get("url", "")

    @url.setter
    def url(self, value: str) -> None:
        if self.supportedInterfaces:
            self.supportedInterfaces[0].url = value
        else:
            self.supportedInterfaces.append(AgentInterface(url=value))

    # ── Serialization ───────────────────────────────────────────────────────

    def to_dict(self, *, legacy_url: bool = False) -> Dict[str, Any]:
        """
        Serialize to a JSON-ready dict.

        Args:
            legacy_url: Also emit the deprecated top-level ``url`` field for
                consumers that only understand pre-v1.0 cards.
        """
        out: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "supportedInterfaces": [i.to_dict() for i in self.supportedInterfaces],
            "capabilities": self.capabilities.to_dict(),
            "skills": [s.to_dict() for s in self.skills],
            "defaultInputModes": list(self.defaultInputModes),
            "defaultOutputModes": list(self.defaultOutputModes),
        }
        if self.provider:
            out["provider"] = dict(self.provider)
        if self.documentationUrl:
            out["documentationUrl"] = self.documentationUrl
        if self.iconUrl:
            out["iconUrl"] = self.iconUrl
        if self.securitySchemes:
            out["securitySchemes"] = {k: dict(v) for k, v in self.securitySchemes.items()}
        if self.security:
            out["security"] = [dict(s) for s in self.security]
        if self.signatures:
            out["signatures"] = [s.to_dict() for s in self.signatures]
        # Unknown fields parsed from a foreign card are preserved verbatim.
        for k, v in self.extra.items():
            out.setdefault(k, v)
        if legacy_url:
            out["url"] = self.url
        return out

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize to a JSON string (what a server serves at the well-known path)."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_json_bytes(self) -> bytes:
        """Serialize to JSON bytes with UTF-8 encoding."""
        return self.to_json().encode("utf-8")

    # ── Parsing ─────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """
        Parse an Agent Card dict (v1.0, tolerant of legacy v0.2/v0.3 cards).

        Legacy normalization:
            - top-level ``url`` (v0.2/v0.3) → ``supportedInterfaces[0]`` (and
              kept in ``extra`` so the original document round-trips),
              only applied when ``supportedInterfaces`` is absent/empty
            - top-level ``protocolVersion`` → ``supportedInterfaces[0].protocolVersion``
            - top-level ``preferredTransport`` → ``supportedInterfaces[0].protocolBinding``
        """
        known = {
            "name", "description", "version", "supportedInterfaces", "capabilities",
            "skills", "provider", "documentationUrl", "iconUrl", "defaultInputModes",
            "defaultOutputModes", "securitySchemes", "security", "signatures",
        }
        extra = {k: v for k, v in data.items() if k not in known}

        interfaces = [AgentInterface.from_dict(i) for i in data.get("supportedInterfaces", [])
                      if isinstance(i, dict)]

        if not interfaces and data.get("url"):
            # Legacy card: synthesize a v1.0 interface from the flat fields.
            interfaces.append(AgentInterface(
                url=data["url"],
                protocolBinding=data.get("preferredTransport", "JSONRPC"),
                protocolVersion=data.get("protocolVersion", A2A_PROTOCOL_VERSION),
            ))

        skills: List[Any] = []
        for s in data.get("skills", []):
            skills.append(AgentSkill.from_dict(s) if isinstance(s, dict) else s)

        signatures = [AgentCardSignature.from_dict(s) for s in data.get("signatures", [])
                      if isinstance(s, dict)]

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", DEFAULT_CARD_VERSION),
            supportedInterfaces=interfaces,
            capabilities=AgentCapabilities.from_dict(data.get("capabilities") or {}),
            skills=skills,
            provider=data.get("provider"),
            documentationUrl=data.get("documentationUrl"),
            iconUrl=data.get("iconUrl"),
            defaultInputModes=list(data.get("defaultInputModes", ["text"])),
            defaultOutputModes=list(data.get("defaultOutputModes", ["text"])),
            securitySchemes=dict(data.get("securitySchemes", {})),
            security=list(data.get("security", [])),
            signatures=signatures,
            extra=extra,
        )

    @classmethod
    def from_json(cls, text: str) -> "AgentCard":
        """Parse an Agent Card from a JSON string."""
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_bytes(cls, raw: bytes) -> "AgentCard":
        """Parse an Agent Card from JSON bytes (e.g. an HTTP response body)."""
        return cls.from_json(raw.decode("utf-8"))
