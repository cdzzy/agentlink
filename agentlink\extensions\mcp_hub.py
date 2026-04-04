"""
MCP Hub - Multi-server MCP coordination for AgentLink.

This module provides a hub that can manage multiple MCP servers,
allowing AgentLink agents to discover and route tool calls across
different MCP providers.

Inspired by Skill_Seekers MCP server patterns (26 tools support).

Usage:
    from agentlink.extensions.mcp_hub import MCPHub, MCPHubNode

    hub = MCPHub()
    hub.register_server("search", "http://search-mcp:8080")
    hub.register_server("code", "http://code-mcp:8081")

    # Create agent node with hub access
    node = MCPHubNode("assistant", handler=my_handler, mcp_hub=hub)
"""

from __future__ import annotations

import json
from typing import Any, Optional, Dict, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from agentlink.adapters.mcp import MCPAdapter, MCPTool, MCPError, MCPToolError


class MCPHubError(Exception):
    """Base exception for MCP Hub errors."""
    pass


class ServerNotFoundError(MCPHubError):
    """Raised when requested MCP server is not registered."""
    pass


class ToolNotFoundError(MCPHubError):
    """Raised when requested tool is not found on any server."""
    pass


@dataclass
class MCPServerInfo:
    """Information about a registered MCP server."""
    name: str
    url: str
    adapter: MCPAdapter
    tools: List[MCPTool] = field(default_factory=list)
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


class MCPHub:
    """
    Hub for managing multiple MCP server connections.

    Features:
    - Multi-server registration
    - Unified tool discovery
    - Tool routing by tags/categories
    - Health monitoring
    - Tool caching

    Example:
        hub = MCPHub()

        # Register multiple servers
        hub.register_server(
            "search",
            "http://search-mcp:8080",
            tags=["web", "search", "research"]
        )
        hub.register_server(
            "code",
            "http://code-mcp:8081",
            tags=["coding", "development"]
        )

        # Discover all tools across servers
        all_tools = hub.discover_all_tools()

        # Route tool call to correct server
        result = hub.call_tool("web_search", {"query": "AI news"})
    """

    def __init__(self, default_timeout: float = 30.0):
        """
        Initialize MCP Hub.

        Args:
            default_timeout: Default timeout for MCP requests
        """
        self._servers: Dict[str, MCPServerInfo] = {}
        self._tool_index: Dict[str, str] = {}  # tool_name -> server_name
        self._default_timeout = default_timeout

    def register_server(
        self,
        name: str,
        url: str,
        timeout: Optional[float] = None,
        tags: Optional[List[str]] = None,
        auto_discover: bool = True,
    ) -> MCPServerInfo:
        """
        Register an MCP server with the hub.

        Args:
            name: Unique name for this server
            url: MCP server URL
            timeout: Optional custom timeout
            tags: Optional tags for categorization
            auto_discover: Whether to immediately discover tools

        Returns:
            MCPServerInfo for the registered server
        """
        if name in self._servers:
            raise ValueError(f"Server '{name}' is already registered")

        timeout = timeout or self._default_timeout
        adapter = MCPAdapter(url, timeout=timeout)

        info = MCPServerInfo(
            name=name,
            url=url,
            adapter=adapter,
            tags=tags or [],
        )

        self._servers[name] = info

        # Auto-discover tools if enabled
        if auto_discover:
            try:
                adapter.connect()
                tools = adapter.list_tools()
                info.tools = tools

                # Build tool index
                for tool in tools:
                    self._tool_index[tool.name] = name
            except Exception as e:
                print(f"Warning: Failed to connect to MCP server '{name}': {e}")

        return info

    def unregister_server(self, name: str) -> bool:
        """
        Unregister an MCP server.

        Args:
            name: Server name to unregister

        Returns:
            True if server was removed
        """
        if name not in self._servers:
            return False

        # Remove tools from index
        server_tools = [t.name for t in self._servers[name].tools]
        for tool_name in server_tools:
            if self._tool_index.get(tool_name) == name:
                del self._tool_index[tool_name]

        del self._servers[name]
        return True

    def get_server(self, name: str) -> MCPServerInfo:
        """Get server info by name."""
        if name not in self._servers:
            raise ServerNotFoundError(f"MCP server '{name}' not found")
        return self._servers[name]

    def discover_all_tools(self) -> List[Dict[str, Any]]:
        """
        Discover and return all tools from all registered servers.

        Returns:
            List of tool dictionaries with server info
        """
        results = []

        for name, info in self._servers.items():
            if not info.enabled:
                continue

            try:
                # Refresh tool list
                info.tools = info.adapter.list_tools()

                for tool in info.tools:
                    results.append({
                        "name": tool.name,
                        "description": tool.description,
                        "server": name,
                        "server_url": info.url,
                        "tags": info.tags,
                        "parameters": tool.parameters,
                    })

                    # Update index
                    self._tool_index[tool.name] = name

            except Exception as e:
                print(f"Warning: Failed to refresh tools from '{name}': {e}")

        return results

    def find_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Find a tool by name across all servers.

        Args:
            tool_name: Name of the tool to find

        Returns:
            Tool info dict or None if not found
        """
        server_name = self._tool_index.get(tool_name)
        if not server_name:
            return None

        info = self._servers.get(server_name)
        if not info:
            return None

        for tool in info.tools:
            if tool.name == tool_name:
                return {
                    "name": tool.name,
                    "description": tool.description,
                    "server": server_name,
                    "server_url": info.url,
                    "tags": info.tags,
                    "parameters": tool.parameters,
                }

        return None

    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        server_hint: Optional[str] = None,
    ) -> Any:
        """
        Call a tool by name, routing to the correct server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            server_hint: Optional server name hint (faster lookup)

        Returns:
            Tool result

        Raises:
            ToolNotFoundError: If tool is not found on any server
            MCPToolError: If tool call fails
        """
        # Try hint first
        if server_hint and server_hint in self._servers:
            info = self._servers[server_hint]
            try:
                return info.adapter.call_tool(tool_name, arguments)
            except Exception:
                pass  # Fall through to full search

        # Find tool across all servers
        server_name = self._tool_index.get(tool_name)
        if not server_name:
            # Try to refresh discovery
            self.discover_all_tools()
            server_name = self._tool_index.get(tool_name)

        if not server_name:
            raise ToolNotFoundError(
                f"Tool '{tool_name}' not found on any registered MCP server"
            )

        info = self._servers[server_name]
        return info.adapter.call_tool(tool_name, arguments)

    def find_tools_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Find all tools with a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching tool info dicts
        """
        results = []

        for name, info in self._servers.items():
            if not info.enabled:
                continue

            if tag in info.tags:
                for tool in info.tools:
                    results.append({
                        "name": tool.name,
                        "description": tool.description,
                        "server": name,
                        "server_url": info.url,
                        "tags": info.tags,
                    })

        return results

    def health_check(self) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all registered servers.

        Returns:
            Dict mapping server name to health status
        """
        results = {}

        for name, info in self._servers.items():
            status = {
                "enabled": info.enabled,
                "url": info.url,
                "tools_count": len(info.tools),
                "healthy": False,
            }

            try:
                info.adapter.connect()
                status["healthy"] = True
            except Exception as e:
                status["error"] = str(e)

            results[name] = status

        return results

    @property
    def servers(self) -> Dict[str, MCPServerInfo]:
        """Get all registered servers."""
        return self._servers.copy()

    @property
    def tools_count(self) -> int:
        """Get total number of tools across all servers."""
        return len(self._tool_index)


class MCPHubNode:
    """
    AgentNode that integrates with MCP Hub for tool discovery.

    This extends the basic AgentNode with MCP Hub capabilities,
    allowing the agent to automatically discover and use tools
    from multiple MCP servers.

    Example:
        hub = MCPHub()
        hub.register_server("search", "http://search-mcp:8080")
        hub.register_server("code", "http://code-mcp:8081")

        # Create node with hub
        node = MCPHubNode(
            "researcher",
            handler=my_handler,
            mcp_hub=hub,
            auto_discover=True,
        )

        # Node now has access to all hub tools
        result = node.call_mcp_tool("web_search", {"query": "AI"})
    """

    def __init__(
        self,
        agent_id: str,
        handler: Callable,
        mcp_hub: MCPHub,
        auto_discover: bool = True,
        default_server_hint: Optional[str] = None,
    ):
        """
        Initialize MCP Hub-enabled AgentNode.

        Args:
            agent_id: Unique identifier for this agent
            handler: Message handler function
            mcp_hub: MCP Hub instance
            auto_discover: Whether to discover tools on init
            default_server_hint: Default server for tool calls
        """
        self.agent_id = agent_id
        self.handler = handler
        self.mcp_hub = mcp_hub
        self.default_server_hint = default_server_hint
        self._discovered_tools: List[Dict[str, Any]] = []

        if auto_discover:
            self.discover_tools()

    def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover all tools from the hub."""
        self._discovered_tools = self.mcp_hub.discover_all_tools()
        return self._discovered_tools

    def call_mcp_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        server_hint: Optional[str] = None,
    ) -> Any:
        """
        Call an MCP tool through the hub.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            server_hint: Optional server hint

        Returns:
            Tool result
        """
        arguments = arguments or {}
        hint = server_hint or self.default_server_hint

        return self.mcp_hub.call_tool(
            tool_name,
            arguments,
            server_hint=hint,
        )

    def find_tools(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find available tools.

        Args:
            tag: Optional tag to filter by

        Returns:
            List of tool info dicts
        """
        if tag:
            return self.mcp_hub.find_tools_by_tag(tag)
        return self._discovered_tools

    @property
    def available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [t["name"] for t in self._discovered_tools]
