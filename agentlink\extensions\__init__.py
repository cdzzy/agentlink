"""
AgentLink Extensions - Extended functionality beyond the core protocol.

Modules:
- mcp_hub: Multi-server MCP coordination hub
"""

from agentlink.extensions.mcp_hub import (
    MCPHub,
    MCPHubNode,
    MCPServerInfo,
    MCPHubError,
    ServerNotFoundError,
    ToolNotFoundError,
)

__all__ = [
    "MCPHub",
    "MCPHubNode",
    "MCPServerInfo",
    "MCPHubError",
    "ServerNotFoundError",
    "ToolNotFoundError",
]
