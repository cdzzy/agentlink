"""
FastMCP-style Adapter for AgentLink.

Provides a high-level, decorator-based API that mirrors the fastmcp library's
ergonomics — so you can register MCP tools with a simple @server.tool() decorator
and Python type annotations instead of hand-writing JSON Schema.

Why this exists
---------------
The existing MCPServerAdapter requires you to write inputSchema by hand:

    server.register_tool("ping", "Ping the server", {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }, handler)

With FastMCPServer you write:

    @server.tool()
    async def ping(message: str) -> str:
        '''Ping the server'''
        return f"pong: {message}"

Type mapping
------------
Python type → JSON Schema type:
    str        → "string"
    int        → "integer"
    float      → "number"
    bool       → "boolean"
    list       → "array"
    dict       → "object"
    Optional[X]→ "X" (not required)
    Any        → {} (no constraint)

Usage
-----
    from agentlink.adapters.fastmcp_adapter import FastMCPServer

    server = FastMCPServer("my-agent")

    @server.tool()
    async def get_weather(city: str, units: str = "celsius") -> str:
        '''Get weather for a city.'''
        return f"Weather in {city}: 22 {units}"

    @server.tool(name="search_db", description="Search the database")
    async def search(query: str, limit: int = 10) -> list:
        return []

    # Drop-in compatible with create_mcp_app()
    from agentlink.adapters.mcp import create_mcp_app
    app = create_mcp_app(server)

    # Or expose an AgentBus with the decorator style
    from agentlink.adapters.fastmcp_adapter import fast_expose_bus
    server = fast_expose_bus(bus, server_name="bus")
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, Awaitable, Optional, get_type_hints

from agentlink.adapters.mcp import MCPServerAdapter, ToolResult


# ─── Python type → JSON Schema ─────────────────────────────────────────────

_PY_TO_JSON: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    bytes: "string",
}


def _py_type_to_json_schema(py_type: Any) -> dict:
    """Convert a Python type annotation to a minimal JSON Schema fragment."""
    origin = getattr(py_type, "__origin__", None)
    args = getattr(py_type, "__args__", ())

    # Optional[X]  →  Union[X, None]
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _py_type_to_json_schema(non_none[0])
        return {}  # complex union → no constraint

    # list[X] / List[X]
    if origin in (list, typing.List) or py_type is list:
        schema: dict = {"type": "array"}
        if args:
            schema["items"] = _py_type_to_json_schema(args[0])
        return schema

    # dict[K, V] / Dict[K, V]
    if origin in (dict, typing.Dict) or py_type is dict:
        return {"type": "object"}

    # Literal[...] → enum
    if origin is typing.Literal:
        return {"type": "string", "enum": list(args)}

    # Primitives
    json_type = _PY_TO_JSON.get(py_type)
    if json_type:
        return {"type": json_type}

    return {}  # fallback — no constraint


def _build_input_schema(fn: Callable) -> dict:
    """
    Inspect a function's signature and type hints to produce a JSON Schema
    ``inputSchema`` object suitable for MCP ``tools/list``.

    - Parameters with defaults are *not* listed in ``required``.
    - ``self`` / ``cls`` are skipped automatically.
    - ``**kwargs`` is ignored.
    - Return annotation and ``*args`` are skipped.
    """
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties: dict[str, dict] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,  # *args
            inspect.Parameter.VAR_KEYWORD,     # **kwargs
        ):
            continue

        py_type = hints.get(param_name, Any)
        # Strip Optional when deciding requiredness
        is_optional = _is_optional(py_type)

        prop_schema = _py_type_to_json_schema(py_type)

        # Add description from default if it is a string (convention)
        if (
            param.default is not inspect.Parameter.empty
            and isinstance(param.default, str)
            and not prop_schema
        ):
            prop_schema["description"] = param.default

        properties[param_name] = prop_schema

        if param.default is inspect.Parameter.empty and not is_optional:
            required.append(param_name)

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _is_optional(py_type: Any) -> bool:
    """Return True if the type is Optional[X] (Union[X, None])."""
    origin = getattr(py_type, "__origin__", None)
    args = getattr(py_type, "__args__", ())
    return origin is typing.Union and type(None) in args


def _wrap_return(value: Any) -> ToolResult:
    """Coerce any return value from a tool handler into a ToolResult."""
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, str):
        return ToolResult(content=[{"type": "text", "text": value}])
    if isinstance(value, (dict, list, int, float, bool)):
        import json
        return ToolResult(content=[{"type": "text", "text": json.dumps(value, ensure_ascii=False)}])
    return ToolResult(content=[{"type": "text", "text": str(value)}])


# ─── FastMCPServer ───────────────────────────────────────────────────────────

class FastMCPServer(MCPServerAdapter):
    """
    Drop-in replacement for MCPServerAdapter with a decorator-based tool API.

    Inherits all MCPServerAdapter behaviour (JSON-RPC dispatch, resources,
    prompts, session management) and adds:

    - ``@server.tool()`` decorator — auto-infers JSON Schema from type hints
    - ``server.tool_count`` property
    - Compatible with ``create_mcp_app()`` from the mcp module

    Example
    -------
    ::

        server = FastMCPServer("weather-agent")

        @server.tool()
        async def get_weather(city: str, units: str = "celsius") -> str:
            '''Get weather for a city'''
            return f"22 {units} in {city}"

        from agentlink.adapters.mcp import create_mcp_app
        app = create_mcp_app(server)
    """

    def __init__(
        self,
        server_name: str = "fastmcp-server",
        protocol_version: str = MCPServerAdapter.PROTOCOL_VERSION,
    ):
        super().__init__(server_name=server_name, protocol_version=protocol_version)

    # ── Decorator API ────────────────────────────────────────────

    def tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to register an async function as an MCP tool.

        Parameters
        ----------
        name:
            Tool name override. Defaults to the function name.
        description:
            Tool description override. Defaults to the function docstring.

        Usage
        -----
        ::

            @server.tool()
            async def search(query: str, limit: int = 10) -> list:
                '''Search the knowledge base'''
                ...

            @server.tool(name="kb_search", description="Search KB")
            async def search(query: str) -> list:
                ...
        """
        def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable:
            tool_name = name or fn.__name__
            tool_desc = description or (inspect.getdoc(fn) or "")
            input_schema = _build_input_schema(fn)

            async def _wrapped_handler(**kwargs: Any) -> ToolResult:
                result = await fn(**kwargs)
                return _wrap_return(result)

            self.register_tool(tool_name, tool_desc, input_schema, _wrapped_handler)
            return fn  # Return original so it can still be called directly

        return decorator

    # ── Convenience properties ───────────────────────────────────

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tool_registry)

    def __repr__(self) -> str:
        return (
            f"FastMCPServer(name={self.server_name!r}, "
            f"tools={self.tool_count}, "
            f"protocol={self.protocol_version!r})"
        )


# ─── Bus-to-FastMCP Bridge ───────────────────────────────────────────────────

def fast_expose_bus(bus: Any, server_name: str = "bus-mcp") -> FastMCPServer:
    """
    Create a FastMCPServer that exposes an AgentBus using the decorator style.

    Functionally equivalent to ``expose_bus_as_mcp()`` but uses FastMCPServer
    so callers can add extra tools via ``@server.tool()``.

    Args:
        bus: An AgentBus instance.
        server_name: Name for the MCP server.

    Returns:
        Configured FastMCPServer.

    Example
    -------
    ::

        server = fast_expose_bus(bus, "my-bus")

        @server.tool()
        async def custom_action(data: str) -> str:
            return "done"

        app = create_mcp_app(server)
    """
    import json as _json

    server = FastMCPServer(server_name=server_name)

    @server.tool(name="ping", description="Ping the MCP server")
    async def _ping() -> str:
        return "pong"

    @server.tool(name="agents_list", description="List all agents registered on the bus")
    async def _agents_list() -> str:
        if hasattr(bus, "registry") and hasattr(bus.registry, "summary"):
            return _json.dumps(bus.registry.summary())
        return "[]"

    @server.tool(name="bus_stats", description="Get AgentBus statistics")
    async def _bus_stats() -> str:
        if hasattr(bus, "stats"):
            return _json.dumps(bus.stats)
        return "{}"

    @server.tool(name="send_message", description="Send a message to an agent via AgentBus")
    async def _send(to_agent: str, message: str) -> str:
        if hasattr(bus, "send"):
            try:
                reply = bus.send(to_agent, message)
                return str(reply)
            except Exception as exc:
                raise RuntimeError(str(exc)) from exc
        raise RuntimeError("Bus does not support send()")

    return server
