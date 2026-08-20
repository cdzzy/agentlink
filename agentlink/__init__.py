"""
AgentLink - The inter-agent communication protocol.

Like HTTP for web services, AgentLink is the missing protocol layer
that lets agents built with different frameworks talk to each other.

  LangGraph Agent  ──�?
  AutoGen Agent    ──┼──�?AgentLink Bus ──�?Any Agent
  CrewAI Agent     ──�?
  Your Custom Agent ─�?

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

from agentlink.protocol.message import (
    AgentMessage,
    MessageType,
    MessageEnvelope,
    AgentAddress,
)
from agentlink.protocol.capability import AgentCapability, CapabilitySet
from agentlink.runtime.node import AgentNode
from agentlink.runtime.bus import AgentBus
from agentlink.runtime.registry import AgentRegistry
from agentlink.schemas import MessageSchema, SchemaRegistry
from agentlink.dlq import DeadLetterQueue, DeadLetter
from agentlink.security import MessageEncryptor, generate_key, encrypt_message, decrypt_message
from agentlink.gateway import ProtocolGateway
from agentlink.transport import WSTransport, WSBridge, serialize_message, deserialize_message
from agentlink.runtime.stream import StreamResult, is_streamable, stream_message

# MCP Adapter (optional dependency)
try:
    from agentlink.adapters.mcp import (
        MCPAdapter,
        MCPAgentNodeMixin,
        MCPTool,
        MCPResource,
        create_mcp_bridge,
        MCPError,
        MCPConnectionError,
        MCPToolError,
    )
    from agentlink.adapters.fastmcp_adapter import FastMCPServer, fast_expose_bus
    _mcp_available = True
except ImportError:
    _mcp_available = False

__version__ = "0.3.0"
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
]

# Add MCP exports if available
if _mcp_available:
    __all__.extend([
        "MCPAdapter",
        "MCPAgentNodeMixin",
        "MCPTool",
        "MCPResource",
        "create_mcp_bridge",
        "MCPError",
        "MCPConnectionError",
        "MCPToolError",
        "FastMCPServer",
        "fast_expose_bus",
    ])
