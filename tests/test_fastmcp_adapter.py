"""
Tests for FastMCPServer — decorator-based MCP tool registration.

Covers:
  - Type inference (_py_type_to_json_schema, _build_input_schema)
  - @server.tool() decorator (name, description, auto-infer)
  - Tool registration and JSON-RPC dispatch
  - Return value coercion (_wrap_return)
  - fast_expose_bus() integration
"""

import json
import pytest
from typing import Optional, List

from agentlink.adapters.fastmcp_adapter import (
    FastMCPServer,
    fast_expose_bus,
    _py_type_to_json_schema,
    _build_input_schema,
    _wrap_return,
)
from agentlink.adapters.mcp import ToolResult


# ─── Type inference ────────────────────────────────────────────────────────────

class TestPyTypeToJsonSchema:
    def test_str(self):
        assert _py_type_to_json_schema(str) == {"type": "string"}

    def test_int(self):
        assert _py_type_to_json_schema(int) == {"type": "integer"}

    def test_float(self):
        assert _py_type_to_json_schema(float) == {"type": "number"}

    def test_bool(self):
        assert _py_type_to_json_schema(bool) == {"type": "boolean"}

    def test_list(self):
        assert _py_type_to_json_schema(list) == {"type": "array"}

    def test_dict(self):
        assert _py_type_to_json_schema(dict) == {"type": "object"}

    def test_optional_str(self):
        schema = _py_type_to_json_schema(Optional[str])
        assert schema == {"type": "string"}

    def test_list_of_str(self):
        from typing import List
        schema = _py_type_to_json_schema(List[str])
        assert schema.get("type") == "array"
        assert schema.get("items") == {"type": "string"}

    def test_unknown_type_fallback(self):
        class Foo:
            pass
        schema = _py_type_to_json_schema(Foo)
        assert schema == {}  # No constraint


class TestBuildInputSchema:
    def test_simple_function(self):
        async def fn(query: str, limit: int = 10) -> list:
            pass

        schema = _build_input_schema(fn)
        assert schema["type"] == "object"
        props = schema["properties"]
        assert props["query"]["type"] == "string"
        assert props["limit"]["type"] == "integer"
        assert "query" in schema["required"]
        assert "limit" not in schema.get("required", [])

    def test_no_params(self):
        async def fn() -> str:
            pass

        schema = _build_input_schema(fn)
        assert schema["properties"] == {}
        assert "required" not in schema

    def test_optional_param_not_required(self):
        async def fn(text: str, tag: Optional[str] = None) -> str:
            pass

        schema = _build_input_schema(fn)
        required = schema.get("required", [])
        assert "text" in required
        assert "tag" not in required

    def test_skips_self_cls(self):
        class Fake:
            async def method(self, x: str) -> str:
                pass

        schema = _build_input_schema(Fake.method)
        assert "self" not in schema["properties"]
        assert "x" in schema["properties"]


# ─── _wrap_return ──────────────────────────────────────────────────────────────

class TestWrapReturn:
    def test_tool_result_passthrough(self):
        r = ToolResult(items=["x"])
        assert _wrap_return(r) is r

    def test_str_return(self):
        result = _wrap_return("hello")
        assert isinstance(result, ToolResult)
        assert result.items[0]["text"] == "hello"

    def test_dict_return(self):
        result = _wrap_return({"key": "value"})
        assert isinstance(result, ToolResult)
        assert "key" in result.items[0]["text"]

    def test_list_return(self):
        result = _wrap_return([1, 2, 3])
        assert isinstance(result, ToolResult)

    def test_int_return(self):
        result = _wrap_return(42)
        assert isinstance(result, ToolResult)
        assert "42" in result.items[0]["text"]


# ─── FastMCPServer ────────────────────────────────────────────────────────────

class TestFastMCPServerInit:
    def test_default_name(self):
        s = FastMCPServer()
        assert s.server_name == "fastmcp-server"

    def test_custom_name(self):
        s = FastMCPServer("my-agent")
        assert s.server_name == "my-agent"

    def test_repr(self):
        s = FastMCPServer("test")
        assert "test" in repr(s)
        assert "tools=" in repr(s)

    def test_tool_count_starts_zero(self):
        s = FastMCPServer()
        assert s.tool_count == 0


class TestFastMCPServerDecorator:
    def test_basic_tool_registration(self):
        s = FastMCPServer("t")

        @s.tool()
        async def greet(name: str) -> str:
            """Greet someone"""
            return f"Hello {name}"

        assert s.tool_count == 1
        tools = s.list_tools()
        assert tools[0]["name"] == "greet"
        assert tools[0]["description"] == "Greet someone"

    def test_tool_name_override(self):
        s = FastMCPServer()

        @s.tool(name="say_hello")
        async def greet(name: str) -> str:
            return f"Hi {name}"

        assert s.list_tools()[0]["name"] == "say_hello"

    def test_tool_description_override(self):
        s = FastMCPServer()

        @s.tool(description="My custom desc")
        async def fn(x: str) -> str:
            """Original docstring"""
            return x

        assert s.list_tools()[0]["description"] == "My custom desc"

    def test_schema_inferred_from_types(self):
        s = FastMCPServer()

        @s.tool()
        async def search(query: str, limit: int = 5) -> list:
            """Search"""
            return []

        tools = s.list_tools()
        schema = tools[0]["inputSchema"]
        assert schema["properties"]["query"]["type"] == "string"
        assert "query" in schema.get("required", [])
        assert "limit" not in schema.get("required", [])

    def test_original_function_still_callable(self):
        """Decorator should not replace the original function."""
        s = FastMCPServer()

        @s.tool()
        async def fn(x: str) -> str:
            return x.upper()

        # fn is still the original coroutine
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(fn("hello"))
        assert result == "HELLO"

    def test_multiple_tools(self):
        s = FastMCPServer()

        @s.tool()
        async def a() -> str:
            return "a"

        @s.tool()
        async def b() -> str:
            return "b"

        assert s.tool_count == 2


# ─── FastMCPServer JSON-RPC dispatch ─────────────────────────────────────────

class TestFastMCPServerDispatch:
    @pytest.mark.asyncio
    async def test_tools_list_returns_registered(self):
        s = FastMCPServer()

        @s.tool()
        async def ping() -> str:
            return "pong"

        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        resp = await s.handle_message(req)
        assert resp is not None
        tools = resp["result"]
        assert any(t["name"] == "ping" for t in tools)

    @pytest.mark.asyncio
    async def test_tool_call_returns_result(self):
        s = FastMCPServer()

        @s.tool()
        async def echo(text: str) -> str:
            """Echo text back"""
            return text

        call_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hello"}},
        }
        resp = await s.handle_message(call_req)
        assert resp is not None
        assert "result" in resp
        assert resp["result"]["content"][0]["text"] == "hello"

    @pytest.mark.asyncio
    async def test_tool_call_str_return_coerced(self):
        s = FastMCPServer()

        @s.tool()
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        resp = await s.handle_message({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "greet", "arguments": {"name": "world"}},
        })
        text = resp["result"]["content"][0]["text"]
        assert "Hello, world!" == text

    @pytest.mark.asyncio
    async def test_tool_call_dict_return_json_encoded(self):
        s = FastMCPServer()

        @s.tool()
        async def info() -> dict:
            return {"version": "1.0", "name": "test"}

        resp = await s.handle_message({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "info", "arguments": {}},
        })
        text = resp["result"]["content"][0]["text"]
        data = json.loads(text)
        assert data["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        s = FastMCPServer()
        resp = await s.handle_message({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
        })
        assert resp is not None
        # Should be an error response (either at JSON-RPC level or tool error)
        # The handler raises MCPToolError which becomes an internal error
        assert "error" in resp or resp.get("result", {}).get("isError", False)

    @pytest.mark.asyncio
    async def test_initialize_handshake(self):
        s = FastMCPServer("test-agent")
        req = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "cursor", "version": "1.0"},
            },
        }
        resp = await s.handle_message(req)
        assert resp["result"]["serverInfo"]["name"] == "test-agent"


# ─── fast_expose_bus ──────────────────────────────────────────────────────────

class TestFastExposeBus:
    def test_returns_fast_mcp_server(self):
        from agentlink.runtime.bus import AgentBus
        bus = AgentBus()
        server = fast_expose_bus(bus, "test-bus")
        assert isinstance(server, FastMCPServer)
        assert server.server_name == "test-bus"

    def test_registers_core_tools(self):
        from agentlink.runtime.bus import AgentBus
        bus = AgentBus()
        server = fast_expose_bus(bus)
        tool_names = {t["name"] for t in server.list_tools()}
        assert "ping" in tool_names
        assert "agents_list" in tool_names
        assert "bus_stats" in tool_names
        assert "send_message" in tool_names

    @pytest.mark.asyncio
    async def test_ping_tool_works(self):
        from agentlink.runtime.bus import AgentBus
        bus = AgentBus()
        server = fast_expose_bus(bus)
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "ping", "arguments": {}},
        })
        assert resp["result"]["content"][0]["text"] == "pong"

    def test_extra_tools_can_be_added(self):
        from agentlink.runtime.bus import AgentBus
        bus = AgentBus()
        server = fast_expose_bus(bus)
        before = server.tool_count

        @server.tool()
        async def custom() -> str:
            return "custom"

        assert server.tool_count == before + 1


# ─── Compatibility: FastMCPServer works with create_mcp_app ───────────────────

class TestFastMCPCompatibility:
    def test_inherits_mcp_server_adapter(self):
        from agentlink.adapters.mcp import MCPServerAdapter
        s = FastMCPServer()
        assert isinstance(s, MCPServerAdapter)

    def test_register_tool_low_level_still_works(self):
        """FastMCPServer should still support the low-level register_tool() API."""
        s = FastMCPServer()

        async def my_handler(**kwargs) -> ToolResult:
            return ToolResult.make([{"type": "text", "text": "ok"}])

        s.register_tool("low_level", "Low level", {}, my_handler)
        assert s.tool_count == 1
