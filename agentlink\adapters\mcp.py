"""
MCP (Model Context Protocol) Adapter for AgentLink.

This adapter allows AgentLink agents to communicate with MCP-compatible tools
and servers, bridging the AgentLink protocol with Anthropic's MCP standard.

Reference: Inspired by claude-mem MCP integration patterns.

Usage:
    from agentlink.adapters.mcp import MCPAdapter
    
    # Create MCP adapter
    mcp = MCPAdapter("http://localhost:37777")
    
    # Discover available tools
    tools = await mcp.discover_tools()
    
    # Call an MCP tool from an AgentLink agent
    result = await mcp.call_tool("search", {"query": "AI news"})
"""

import json
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""
    pass


class MCPToolError(MCPError):
    """Raised when an MCP tool call fails."""
    pass


@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    parameters: dict
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class MCPResource:
    """Represents an MCP resource."""
    uri: str
    name: str
    mime_type: str
    description: Optional[str] = None


class MCPAdapter:
    """
    Adapter for MCP (Model Context Protocol) integration.
    
    This adapter allows AgentLink agents to:
    - Discover and call MCP tools
    - Access MCP resources
    - Handle MCP prompts
    
    Example:
        adapter = MCPAdapter("http://localhost:37777")
        await adapter.connect()
        
        # List available tools
        tools = await adapter.list_tools()
        
        # Call a tool
        result = await adapter.call_tool("search", {"query": "Python"})
    """
    
    def __init__(self, server_url: str, timeout: float = 30.0):
        """
        Initialize MCP adapter.
        
        Args:
            server_url: Base URL of the MCP server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._tools: list[MCPTool] = []
        self._connected = False
        
    def _http_request(self, method: str, path: str, data: Optional[dict] = None) -> dict:
        """
        Make HTTP request to MCP server.
        
        Note: This is a synchronous implementation using urllib.
        For async usage, subclass and override with aiohttp or httpx.
        """
        import urllib.request
        import urllib.error
        
        url = f"{self.server_url}{path}"
        headers = {"Content-Type": "application/json"}
        
        if data:
            body = json.dumps(data).encode("utf-8")
        else:
            body = None
        
        req = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise MCPToolError(f"MCP request failed: {e.code} - {error_body}")
        except urllib.error.URLError as e:
            raise MCPConnectionError(f"Failed to connect to MCP server: {e.reason}")
    
    def connect(self) -> bool:
        """
        Test connection to MCP server.
        
        Returns:
            True if connection successful
            
        Raises:
            MCPConnectionError: If connection fails
        """
        try:
            # Try to access the server root or health endpoint
            self._http_request("GET", "/")
            self._connected = True
            return True
        except MCPConnectionError:
            raise
        except Exception as e:
            # Some servers may not have a root endpoint
            # Try to discover tools as a connectivity test
            try:
                self.list_tools()
                self._connected = True
                return True
            except Exception:
                raise MCPConnectionError(f"Failed to connect to MCP server: {e}")
    
    def list_tools(self) -> list[MCPTool]:
        """
        Discover available tools from the MCP server.
        
        Returns:
            List of available tools
        """
        try:
            response = self._http_request("GET", "/tools")
            tools_data = response.get("tools", [])
            
            self._tools = [
                MCPTool(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("parameters", {}),
                )
                for t in tools_data
            ]
            return self._tools
        except Exception as e:
            # Fallback: return empty list if endpoint doesn't exist
            return []
    
    def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result
            
        Raises:
            MCPToolError: If tool call fails
        """
        try:
            response = self._http_request(
                "POST",
                f"/tools/{tool_name}",
                {"arguments": arguments},
            )
            return response.get("result")
        except MCPToolError:
            raise
        except Exception as e:
            raise MCPToolError(f"Tool call failed: {e}")
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None
    
    def to_agentlink_capability(self, tool: MCPTool) -> dict:
        """
        Convert MCP tool to AgentLink capability format.
        
        Returns:
            Capability dict compatible with AgentLink
        """
        return {
            "name": f"mcp:{tool.name}",
            "description": tool.description,
            "type": "external_tool",
            "mcp_tool": tool.name,
            "parameters": tool.parameters,
        }


class MCPAgentNodeMixin:
    """
    Mixin for AgentNode to add MCP capabilities.
    
    Usage:
        class MyAgentNode(AgentNode, MCPAgentNodeMixin):
            def __init__(self, *args, mcp_adapter=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.mcp = mcp_adapter
    """
    
    def __init__(self, *args, mcp_adapter: Optional[MCPAdapter] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcp = mcp_adapter
        self._mcp_tools: dict[str, MCPTool] = {}
    
    def discover_mcp_tools(self) -> list[MCPTool]:
        """Discover and cache MCP tools."""
        if self.mcp:
            tools = self.mcp.list_tools()
            self._mcp_tools = {t.name: t for t in tools}
            return tools
        return []
    
    def call_mcp_tool(self, tool_name: str, **kwargs) -> Any:
        """Call an MCP tool by name."""
        if not self.mcp:
            raise MCPError("No MCP adapter configured")
        return self.mcp.call_tool(tool_name, kwargs)


def create_mcp_bridge(agent_node, mcp_url: str) -> MCPAdapter:
    """
    Create an MCP bridge for an AgentLink agent node.
    
    This allows the agent to use MCP tools as if they were native capabilities.
    
    Args:
        agent_node: The AgentNode to bridge
        mcp_url: URL of the MCP server
        
    Returns:
        Configured MCPAdapter
    """
    adapter = MCPAdapter(mcp_url)
    adapter.connect()
    
    # Discover tools and add as capabilities
    tools = adapter.list_tools()
    for tool in tools:
        capability = adapter.to_agentlink_capability(tool)
        # Add to agent's capabilities if the node supports it
        if hasattr(agent_node, 'add_capability'):
            agent_node.add_capability(capability)
    
    return adapter
