"""
MCP (Model Context Protocol) Adapter for AgentLink.

This adapter allows AgentLink agents to communicate with MCP-compatible tools
and servers, bridging the AgentLink protocol with Anthropic's MCP standard.

Supports both:
- **MCP Client (MCPAdapter)**: Connect to external MCP servers, discover and call tools
- **MCP Server (MCPServerAdapter)**: Expose AgentLink as an MCP server endpoint

Reference: https://modelcontextprotocol.io / 2025-11-25 spec

Usage (Client):
    from agentlink.adapters.mcp import MCPAdapter
    mcp = MCPAdapter("http://localhost:37777")
    tools = await mcp.list_tools()
    result = mcp.call_tool("search", {"query": "AI news"})

Usage (Server):
    from agentlink.adapters.mcp import MCPServerAdapter, expose_bus_as_mcp
    server = MCPServerAdapter(server_name="my-agent")
    server.register_tool("search", "Search data", {}, my_handler)
    app = create_mcp_app(server)
"""

from __future__ import annotations

import json
import uuid
import asyncio
from typing import Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import IntEnum


# ─── Exceptions ────────────────────────────────────────────────

class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""
    pass


class MCPToolError(MCPError):
    """Raised when an MCP tool call fails."""
    pass


# ─── Data Classes ──────────────────────────────────────────────

@dataclass
class MCPTool:
    """Represents an MCP tool definition."""
    name: str
    description: str
    parameters: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }


@dataclass
class MCPResource:
    """Represents an MCP resource."""
    uri: str
    name: str
    mime_type: str
    description: Optional[str] = None


class ToolResult:
    """Result of a tool call.

    Provides both a data attribute and factory class-methods with the same
    ``content`` name by using a regular (non-dataclass) class.

    Factory usage::

        r = ToolResult.content([{"type": "text", "text": "ok"}])
        r = ToolResult.error("Something went wrong")

    Direct construction (also supported for backward compat)::

        r = ToolResult(content=[...], is_error=False)
    """

    def __init__(
        self,
        content: "list[Any] | None" = None,
        is_error: bool = False,
        # Alias used internally (fastmcp_adapter uses items=)
        items: "list[Any] | None" = None,
    ) -> None:
        # Accept either `content=` or `items=` kwarg; content takes priority
        self._content: list[Any] = content if content is not None else (items or [])
        self.is_error = is_error

    # ── Public data attribute ────────────────────────────────────

    @property
    def content(self) -> "list[Any]":  # type: ignore[override]
        """The list of content items (wire format)."""
        return self._content

    @content.setter
    def content(self, value: "list[Any]") -> None:
        self._content = value

    # Alias kept for internal code in fastmcp_adapter that uses .items
    @property
    def items(self) -> "list[Any]":
        return self._content

    # ── Factory class-methods ────────────────────────────────────

    @classmethod
    def make(cls, data: list) -> "ToolResult":
        """Create a successful ToolResult from a list of content items."""
        return cls(content=data, is_error=False)

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        """Create an error ToolResult with a text message."""
        return cls(content=[{"type": "text", "text": message}], is_error=True)

    # ── Serialisation ────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "content": self._content,
            "isError": self.is_error,
        }

    def __str__(self) -> str:
        if self.is_error:
            return f"ToolError({self._content})"
        return f"ToolResult({len(self._content)} items)"

    def __repr__(self) -> str:
        return f"ToolResult(content={self._content!r}, is_error={self.is_error})"

    # ── Equality (useful in tests) ───────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolResult):
            return NotImplemented
        return self._content == other._content and self.is_error == other.is_error


# Patch: ToolResult.content(data) as a class-method while .content on instances
# remains the property above.  We cannot have both at the class level with the
# same name, so we implement the factory via a descriptor shim.

class _ContentDescriptor:
    """
    Descriptor that returns the instance property when accessed on an instance,
    and acts as a class-method factory when called on the class.
    """

    def __get__(self, obj: "ToolResult | None", objtype: type) -> "Any":
        if obj is None:
            # Class-level access: return a factory callable
            return lambda data: ToolResult(content=data, is_error=False)
        # Instance-level access: return the content list
        return obj._content

    def __set__(self, obj: "ToolResult", value: "list[Any]") -> None:
        obj._content = value


ToolResult.content = _ContentDescriptor()  # type: ignore[assignment]


# ─── JSON-RPC Types ────────────────────────────────────────────

@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 request object."""
    method: str
    params: Any = None
    req_id: Optional[int | str] = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict:
        d = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params is not None:
            d["params"] = self.params
        if self.req_id is not None:
            d["id"] = self.req_id
        return d


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 response object."""
    result: Any = None
    error: Optional[dict] = None
    resp_id: Optional[int | str] = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc}
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        if self.resp_id is not None:
            d["id"] = self.resp_id
        return d


class McpErrorCode(IntEnum):
    """Standard MCP error codes per the specification."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # MCP-specific
    SERVER_NOT_INITIALIZED = -32002
    OVERLOADED = -32029


# ─── Tool Name Conversion ──────────────────────────────────────

def _agent_to_tool_name(agent_name: str) -> str:
    """
    Convert an AgentLink agent-style name to MCP tool name.

    Converts colons to double underscores for safe transport.
    Example: "agent:db:query" -> "agent__db__query"
    """
    return agent_name.replace(":", "__")


def _parse_tool_name(tool_name: str) -> tuple[str, str]:
    """
    Parse an MCP tool name back into namespace + name parts.

    Returns: (namespace, local_name)
    Example: "agent__db__query" -> ("agent:db", "query")
    """
    if "__" in tool_name:
        parts = tool_name.split("__", 1)
        ns = parts[0].replace(":", "_")  # Restore colon convention
        return ns, parts[1]
    return "", tool_name


# ─── MCP Server Adapter ────────────────────────────────────────

class MCPServerAdapter:
    """
    Full MCP Server implementation following the 2025-11-25 spec.

    This adapter exposes AgentLink capabilities as an MCP-compliant server,
    allowing any MCP client to discover tools, call them, access resources,
    etc.

    Features:
    - JSON-RPC 2.0 over Streamable HTTP
    - Tool registration and invocation
    - Resource management (read-only)
    - Prompt templates
    - Session management with UUID-based session IDs
    - Logging level control
    - Initialize handshake with version negotiation
    """

    PROTOCOL_VERSION = "2025-11-25"

    def __init__(
        self,
        server_name: str = "mcp-server",
        protocol_version: str = PROTOCOL_VERSION,
    ):
        self.server_name = server_name
        self.protocol_version = protocol_version
        self._tool_registry: dict[str, Callable[..., Awaitable[ToolResult]]] = {}
        self._tool_schemas: dict[str, dict] = {}
        self._resource_registry: dict[str, dict] = {}  # uri -> {name, mime_type, ...}
        self._prompt_registry: dict[str, dict] = {}
        self.session_id: Optional[str] = None
        self._initialized = False
        self._logging_level: str = "info"
        self.raw_send: Optional[Callable] = None  # Hook for sending responses

    # ── Session Management ──────────────────────────────────

    def _generate_session_id(self) -> str:
        return str(uuid.uuid4())

    def _ensure_session(self) -> str:
        if not self.session_id:
            self.session_id = self._generate_session_id()
        return self.session_id

    # ── Initialize Handshake ────────────────────────────────

    async def _handle_initialize(self, params: dict) -> dict:
        """Handle the initialize request — negotiate protocol version."""
        client_version = params.get("protocolVersion", "2025-11-25")
        client_info = params.get("clientInfo", {})
        # Use the lower of client/server version (simplified)
        negotiated_version = min(client_version, self.protocol_version)
        self._ensure_session()
        self._initialized = True
        return {
            "protocolVersion": negotiated_version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
                "logging": {},
            },
            "serverInfo": {
                "name": self.server_name,
                "version": "0.1.0",
            },
        }

    # ── Message Dispatcher ──────────────────────────────────

    async def handle_message(self, message: dict) -> Optional[dict]:
        """
        Route an incoming JSON-RPC message to the appropriate handler.

        Args:
            message: Parsed JSON-RPC request/notification dictionary

        Returns:
            Response dict for requests, None for notifications
        """
        method = message.get("method", "")
        params = message.get("params", {})
        req_id = message.get("id")

        try:
            handler = self._get_handler(method)
            if handler:
                result = await handler(params)
                if req_id is not None:
                    return {"jsonrpc": "2.0", "result": result, "id": req_id}
                return None  # Notification

            # Unknown method
            if req_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": McpErrorCode.METHOD_NOT_FOUND,
                        "message": f"Method not found: {method}",
                    },
                    "id": req_id,
                }
            return None

        except Exception as e:
            if req_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": McpErrorCode.INTERNAL_ERROR,
                        "message": str(e),
                    },
                    "id": req_id,
                }
            return None

    def _get_handler(self, method: str) -> Optional[Callable]:
        handlers = {
            "initialize": self._handle_initialize_wrapper,
            "notifications/initialized": self._handle_initialized_notification,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
            "logging/setLevel": self._handle_set_logging_level,
        }
        return handlers.get(method)

    # ── Protocol Method Handlers ────────────────────────────

    async def _handle_initialize_wrapper(self, params: dict) -> dict:
        return await self._initialize(params)

    async def _initialize(self, params: dict) -> dict:
        return await self._handle_initialize(params)

    async def _handle_initialized_notification(self, params: dict) -> None:
        self._initialized = True
        return None

    async def _handle_ping(self, params: dict) -> dict:
        return {}

    async def _handle_tools_list(self, params: dict) -> list:
        cursor = params.get("cursor")
        tools = [
            {
                "name": name,
                "description": schema.get("description", ""),
                "inputSchema": schema.get("inputSchema", {}),
            }
            for name, schema in self._tool_schemas.items()
        ]
        if cursor:
            # Simple pagination: skip items before cursor
            try:
                idx = int(cursor)
                tools = tools[idx:]
            except (ValueError, TypeError):
                pass
        return tools

    async def _handle_tools_call(self, params: dict) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        handler = self._tool_registry.get(name)
        if not handler:
            raise MCPToolError(f"Tool not found: {name}")

        result = await handler(**arguments)
        if isinstance(result, ToolResult):
            return result.to_dict()
        return {"content": [str(result)], "isError": False}

    async def _handle_resources_list(self, params: dict) -> list:
        return [
            {"uri": uri, **info}
            for uri, info in self._resource_registry.items()
        ]

    async def _handle_resources_read(self, params: dict) -> list:
        uri = params.get("uri", "")
        resource = self._resource_registry.get(uri)
        if not resource:
            raise MCPError(f"Resource not found: {uri}")
        contents = resource.get("contents", "")
        mime_type = resource.get("mime_type", resource.get("mimeType", "text/plain"))
        return [{
            "uri": uri,
            "mimeType": mime_type,
            "text": contents,
            "type": "text",
        }]

    async def _handle_prompts_list(self, params: dict) -> list:
        return list(self._prompt_registry.values())

    async def _handle_prompts_get(self, params: dict) -> dict:
        name = params.get("name", "")
        prompt = self._prompt_registry.get(name)
        if not prompt:
            raise MCPError(f"Prompt not found: {name}")
        return prompt

    async def _handle_set_logging_level(self, params: dict) -> dict:
        level = params.get("level", "info")
        valid_levels = ("debug", "info", "notice", "warning", "error", "critical", "off")
        if level in valid_levels:
            self._logging_level = level
        return {}

    # ── Public Registration API ─────────────────────────────

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: Callable[..., Awaitable[ToolResult]],
    ) -> None:
        """Register a callable as an MCP tool."""
        self._tool_registry[name] = handler
        self._tool_schemas[name] = {
            "description": description,
            "inputSchema": input_schema,
        }

    def register_resource(
        self,
        uri: str,
        name: str,
        mime_type: str,
        contents: str = "",
        description: Optional[str] = None,
    ) -> None:
        """Register a readable resource."""
        self._resource_registry[uri] = {
            "name": name,
            "mimeType": mime_type,
            "mimeType": mime_type,
            "contents": contents,
            "description": description or "",
        }

    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: Optional[list[dict]] = None,
    ) -> None:
        """Register a prompt template."""
        self._prompt_registry[name] = {
            "name": name,
            "description": description,
            "arguments": arguments or [],
        }

    # ── Query API ────────────────────────────────────────────

    def list_tools(self) -> list[dict]:
        """Return registered tool definitions."""
        return [
            {"name": name, **schema}
            for name, schema in self._tool_schemas.items()
        ]

    def list_resources(self) -> list[dict]:
        """Return registered resources."""
        return [{"uri": uri, **info} for uri, info in self._resource_registry.items()]

    def read_resource(self, uri: str) -> Optional[list]:
        """Read a resource by URI. Returns content list or None."""
        resource = self._resource_registry.get(uri)
        if resource:
            return [{"type": "text", "text": resource.get("contents", "")}]
        return None

    def list_prompts(self) -> list[dict]:
        """Return registered prompts."""
        return list(self._prompt_registry.values())


# ─── MCP Client Adapter ─────────────────────────────────────────

class MCPAdapter:
    """
    Client-side adapter for connecting to external MCP servers.

    Allows AgentLink agents to discover and call tools on remote MCP servers.

    Example:
        adapter = MCPAdapter("http://localhost:8000")
        adapter.connect()

        tools = adapter.list_tools()
        result = adapter.call_tool("search", {"query": "Python"})
    """

    def __init__(self, server_url: str, timeout: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._tools: list[MCPTool] = []
        self._connected = False
        self._server_capabilities: dict = {}

    def _http_request(self, method: str, path: str, data: Optional[dict] = None) -> dict:
        """Make HTTP request to MCP server using urllib (synchronous)."""
        import urllib.request
        import urllib.error

        url = f"{self.server_url}{path}"
        headers = {"Content-Type": "application/json"}

        body = json.dumps(data).encode("utf-8") if data else None

        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise MCPToolError(f"MCP request failed: {e.code} - {error_body}")
        except urllib.error.URLError as e:
            raise MCPConnectionError(f"Failed to connect to MCP server: {e.reason}")

    def connect(self) -> bool:
        """Test connection to MCP server. Raises MCPConnectionError on failure."""
        try:
            self._http_request("GET", "/")
            self._connected = True
            return True
        except MCPConnectionError:
            raise
        except Exception as e:
            try:
                self.list_tools()
                self._connected = True
                return True
            except Exception:
                raise MCPConnectionError(f"Failed to connect to MCP server: {e}")

    def list_tools(self) -> list[MCPTool]:
        """Discover available tools from the MCP server."""
        try:
            response = self._http_request("GET", "/tools")
            tools_data = response.get("tools", [])
            self._tools = [
                MCPTool(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    parameters=t.get("parameters", t.get("inputSchema", {})),
                )
                for t in tools_data
            ]
            return self._tools
        except Exception:
            return []

    def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call an MCP tool by name."""
        try:
            response = self._http_request(
                "POST", f"/tools/{tool_name}", {"arguments": arguments},
            )
            return response.get("result")
        except MCPToolError:
            raise
        except Exception as e:
            raise MCPToolError(f"Tool call failed: {e}")

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a cached tool by name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    def to_agentlink_capability(self, tool: MCPTool) -> dict:
        """Convert MCP tool to AgentLink capability format."""
        return {
            "name": f"mcp:{tool.name}",
            "description": tool.description,
            "type": "external_tool",
            "mcp_tool": tool.name,
            "parameters": tool.parameters,
        }

    @staticmethod
    def parse_server_capabilities(caps: dict) -> dict:
        """Parse and normalize server capabilities response."""
        return {
            "tools": caps.get("tools", {}),
            "resources": caps.get("resources", {}),
            "prompts": caps.get("prompts", {}),
            "logging": caps.get("logging", {}),
            "experimental": caps.get("experimental", {}),
        }


# ─── MCP Agent Node Mixin ──────────────────────────────────────

class MCPAgentNodeMixin:
    """
    Mixin class that adds MCP client capability to AgentNode.

    Usage:
        class MyAgent(AgentNode, MCPAgentNodeMixin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.mcp = MCPAdapter("http://mcp-server:8000")

        agent.discover_mcp_tools()   # Find available MCP tools
        agent.call_mcp_tool("search", query="AI news")  # Call a tool
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcp = getattr(kwargs.pop('mcp_adapter', None), None)
        self._mcp_tools: dict[str, MCPTool] = {}

    def discover_mcp_tools(self) -> list[MCPTool]:
        """Discover and cache MCP tools from connected server."""
        mcp = getattr(self, "mcp", None)
        if mcp:
            tools = mcp.list_tools()
            self._mcp_tools = {t.name: t for t in tools}
            return tools
        return []

    def call_mcp_tool(self, tool_name: str, **kwargs) -> Any:
        """Call an MCP tool by name through the configured adapter."""
        mcp = getattr(self, "mcp", None)
        if not mcp:
            raise MCPError("No MCP adapter configured")
        return mcp.call_tool(tool_name, kwargs)


# ─── HTTP App Factory ──────────────────────────────────────────

def create_mcp_app(server: MCPServerAdapter):
    """
    Create an aiohttp web application serving the given MCPServerAdapter.

    All POST requests to / are treated as JSON-RPC calls.
    GET / returns basic server info.
    """
    from aiohttp import web

    async def handle_rpc(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {"code": McpErrorCode.PARSE_ERROR, "message": "Invalid JSON"},
                "id": None,
            }, status=400)

        result = await server.handle_message(body)
        if result is None:
            return web.Response(status=202)  # Accepted (notification)
        return web.json_response(result)

    async def handle_health(request: web.Request) -> web.Response:
        return web.json_response({
            "name": server.server_name,
            "version": "0.1.0",
            "protocol": server.protocol_version,
            "status": "ok",
        })

    app = web.Application()
    app.router.add_post("/", handle_rpc)
    app.router.add_get("/", handle_health)
    return app


# ─── Bus-to-MCP Bridge ──────────────────────────────────────────

def expose_bus_as_mcp(bus, server_name: str = "bus-mcp") -> MCPServerAdapter:
    """
    Create an MCPServerAdapter that exposes an AgentBus as MCP tools.

    Registers bus operations (send, ping, stats, list_agents) as MCP tools.
    Returns the configured server ready for use with create_mcp_app().

    Args:
        bus: An AgentBus instance
        server_name: Name for the MCP server

    Returns:
        Configured MCPServerAdapter
    """
    import asyncio as _asyncio

    server = MCPServerAdapter(server_name=server_name)

    async def mcp_ping(**kwargs):
        return ToolResult.make([{"type": "text", "text": "pong"}])

    async def mcp_list_agents(**kwargs):
        if hasattr(bus, 'registry') and hasattr(bus.registry, 'summary'):
            summary = bus.registry.summary()
            return ToolResult.make([{"type": "text", "text": json.dumps(summary)}])
        return ToolResult.make([{"type": "text", "text": "[]"}])

    async def mcp_bus_stats(**kwargs):
        if hasattr(bus, 'stats'):
            return ToolResult.make([{"type": "text", "text": json.dumps(bus.stats)}])
        return ToolResult.make([{"type": "text", "text": "{}"}])

    async def mcp_send(to_agent: str, message: str, **kwargs):
        if hasattr(bus, 'send'):
            try:
                reply = bus.send(to_agent, message)
                return ToolResult.make([{"type": "text", "text": str(reply)}])
            except Exception as e:
                return ToolResult.error(str(e))
        return ToolResult.error("Bus does not support send()")

    server.register_tool("ping", "Ping the MCP server", {}, mcp_ping)
    server.register_tool("agents/list", "List all registered agents", {
        "type": "object",
        "properties": {},
    }, mcp_list_agents)
    server.register_tool("stats", "Get bus statistics", {
        "type": "object",
        "properties": {},
    }, mcp_bus_stats)
    server.register_tool("send", "Send a message via AgentBus", {
        "type": "object",
        "properties": {
            "to_agent": {"type": "string", "description": "Target agent address"},
            "message": {"type": "string", "description": "Message content"},
        },
        "required": ["to_agent", "message"],
    }, mcp_send)

    return server


# ─── Bridge factory function (legacy alias) ────────────────────

def create_mcp_bridge(agent_node, mcp_url: str) -> MCPAdapter:
    """
    Create an MCP bridge for an AgentLink agent node.

    Discovers tools and registers them as AgentLink capabilities.
    """
    adapter = MCPAdapter(mcp_url)
    adapter.connect()

    tools = adapter.list_tools()
    for tool in tools:
        capability = adapter.to_agentlink_capability(tool)
        if hasattr(agent_node, 'add_capability'):
            agent_node.add_capability(capability)

    return adapter
