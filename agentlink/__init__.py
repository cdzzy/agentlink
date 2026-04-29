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
    _mcp_available = True
except ImportError:
    _mcp_available = False

__version__ = "0.1.0"
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
    ])
