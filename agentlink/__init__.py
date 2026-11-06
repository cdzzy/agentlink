"""
AgentLink - The inter-agent communication protocol.

Like HTTP for web services, AgentLink is the missing protocol layer
that lets agents built with different frameworks talk to each other.

  LangGraph Agent  ──┐
  AutoGen Agent    ──┼──► AgentLink Bus ──► Any Agent
  CrewAI Agent     ──┘
  Your Custom Agent ─┘

Usage:
    from agentlink import AgentNode, AgentBus, AgentMessage

    # Wrap any agent
    node = AgentNode("my-agent", handler=my_agent_fn)

    # Connect to the bus
    bus = AgentBus()
    bus.register(node)

    # Send a message to any other agent
    reply = node.send("other-agent", "What is 2+2?")
"""

from agentlink.a2a import (
    A2A_PROTOCOL_VERSION,
    ENVELOPE_METADATA_KEY,
    A2ATransport,
    A2ATransportError,
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    a2a_message_to_agent_message,
    a2a_task_to_agent_message,
    agent_card_url,
    agent_message_to_a2a_message,
    agent_message_to_a2a_task,
)
from agentlink.a2a.card import AGENT_CARD_WELLKNOWN_PATH
from agentlink.adapters.ag_ui import (
    AGUI_APPROVAL_SCHEMA,
    AGUI_PROTOCOL_VERSION,
    AGUIEmitter,
    AGUIError,
    AGUIEventType,
    AGUIInterrupt,
    AGUIInterruptManager,
    AGUIResume,
    AGUIResumeError,
    create_ag_ui_app,
    expose_node_as_ag_ui,
    last_user_text,
)
from agentlink.dlq import DeadLetter, DeadLetterQueue
from agentlink.gateway import ProtocolGateway
from agentlink.hub import HubClient, HubRegistry, HubServer
from agentlink.integrations.engram import (
    EngramMCPClient,
    EngramMCPError,
    EngramMemoryBackend,
    attach_memory,
)
from agentlink.protocol.capability import AgentCapability, CapabilitySet
from agentlink.protocol.message import (
    AgentAddress,
    AgentMessage,
    MessageEnvelope,
    MessageType,
)
from agentlink.runtime.bus import AgentBus
from agentlink.runtime.node import AgentNode
from agentlink.runtime.registry import AgentRegistry
from agentlink.runtime.stream import StreamResult, is_streamable, stream_message
from agentlink.schemas import MessageSchema, SchemaRegistry
from agentlink.security import MessageEncryptor, decrypt_message, encrypt_message, generate_key
from agentlink.tracing import InMemorySpanExporter, SpanRecord, instrument_bus
from agentlink.transport import WSBridge, WSTransport, deserialize_message, serialize_message

# MCP Adapter (optional dependency)
try:
    from agentlink.adapters.fastmcp_adapter import FastMCPServer, fast_expose_bus  # noqa: F401
    from agentlink.adapters.mcp import (  # noqa: F401
        MCP_PROTOCOL_HEADER,
        MCP_SESSION_HEADER,
        MCPAdapter,
        MCPAgentNodeMixin,
        MCPConnectionError,
        MCPError,
        MCPResource,
        MCPServerAdapter,
        MCPTool,
        MCPToolError,
        create_mcp_app,
        create_mcp_bridge,
        create_streamable_mcp_app,
        expose_bus_as_mcp,
    )
    _mcp_available = True
except ImportError:
    _mcp_available = False

# A2A v1.0 card verification (optional dependency: pip install cdzzy-agentlink[a2a])
try:
    from agentlink.a2a import (  # noqa: F401
        A2AVerificationError,
        generate_signing_key,
        sign_agent_card,
        verify_agent_card,
    )
    _a2a_verify_available = True
except ImportError:
    _a2a_verify_available = False

def _version_from_pyproject() -> str:
    """Read the canonical version from the adjacent pyproject.toml (PEP 621)."""
    try:
        import re
        from pathlib import Path

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        match = re.search(
            r'^version\s*=\s*"([^"]+)"',
            pyproject.read_text(encoding="utf-8"),
            re.M,
        )
        return match.group(1) if match else ""
    except Exception:  # noqa: BLE001
        return ""


def _version_from_metadata() -> str:
    """Fall back to the installed distribution metadata (regular installs)."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("cdzzy-agentlink")
    except Exception:  # noqa: BLE001
        return ""


# pyproject.toml is the single source of truth in a source checkout / editable
# install, so __version__ can never drift from the declared version; regular
# installed copies (no adjacent pyproject.toml) fall back to dist metadata.
__version__ = _version_from_pyproject() or _version_from_metadata() or "0.0.0"
__all__ = [
    "AgentMessage",
    "MessageType",
    "MessageEnvelope",
    "AgentAddress",
    "AgentCapability",
    "CapabilitySet",
    "AgentNode",
    "AgentBus",
    "AgentRegistry",
    "MessageSchema",
    "SchemaRegistry",
    "DeadLetterQueue",
    "DeadLetter",
    "MessageEncryptor",
    "generate_key",
    "encrypt_message",
    "decrypt_message",
    "ProtocolGateway",
    "WSTransport",
    "WSBridge",
    "serialize_message",
    "deserialize_message",
    "StreamResult",
    "is_streamable",
    "stream_message",
    "instrument_bus",
    "InMemorySpanExporter",
    "SpanRecord",
    "EngramMCPClient",
    "EngramMemoryBackend",
    "attach_memory",
    "EngramMCPError",
    "HubServer",
    "HubClient",
    "HubRegistry",
    # A2A v1.0 compatibility layer
    "A2A_PROTOCOL_VERSION",
    "AGENT_CARD_WELLKNOWN_PATH",
    "AgentCard",
    "AgentCapabilities",
    "AgentInterface",
    "AgentSkill",
    "A2ATransport",
    "A2ATransportError",
    "agent_card_url",
    "agent_message_to_a2a_message",
    "a2a_message_to_agent_message",
    "agent_message_to_a2a_task",
    "a2a_task_to_agent_message",
    "ENVELOPE_METADATA_KEY",
    # AG-UI v1.0 (agent → human)
    "AGUI_PROTOCOL_VERSION",
    "AGUIEventType",
    "AGUIEmitter",
    "AGUIInterrupt",
    "AGUIInterruptManager",
    "AGUIResume",
    "AGUI_APPROVAL_SCHEMA",
    "AGUIError",
    "AGUIResumeError",
    "create_ag_ui_app",
    "expose_node_as_ag_ui",
    "last_user_text",
]

# Add A2A card verification exports if the optional extras are installed
if _a2a_verify_available:
    __all__.extend([
        "A2AVerificationError",
        "generate_signing_key",
        "sign_agent_card",
        "verify_agent_card",
    ])

# Add MCP exports if available
if _mcp_available:
    __all__.extend([
        "MCPAdapter",
        "MCPAgentNodeMixin",
        "MCPServerAdapter",
        "MCPTool",
        "MCPResource",
        "create_mcp_bridge",
        "create_mcp_app",
        "create_streamable_mcp_app",
        "expose_bus_as_mcp",
        "MCP_SESSION_HEADER",
        "MCP_PROTOCOL_HEADER",
        "MCPError",
        "MCPConnectionError",
        "MCPToolError",
        "FastMCPServer",
        "fast_expose_bus",
    ])
