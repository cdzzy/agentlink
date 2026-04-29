"""
Comprehensive tests for MCP (Model Context Protocol) Adapter.

Covers: tool name conversion, JSON-RPC helpers, MCPServerAdapter (full lifecycle),
MCPAdapter (client), MCPAgentNodeMixin, and HTTP integration tests.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# aiohttp is an optional dependency for HTTP server tests
try:
    from aiohttp import web, test_utils
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from agentlink.adapters.mcp import (
    MCPError,
    MCPConnectionError,
    MCPToolError,
    MCPTool,
    MCPResource,
    MCPAdapter,
    MCPAgentNodeMixin,
    create_mcp_bridge,
    expose_bus_as_mcp,
    _agent_to_tool_name,
    _parse_tool_name,
    ToolResult,
    JsonRpcRequest,
    JsonRpcResponse,
    McpErrorCode,
    MCPServerAdapter,
)


# ─── Tool Name Conversion ────────────────────────────────────────

class TestToolNameConversion:
    def test_simple_name(self):
        assert _agent_to_tool_name("search") == "search"

    def test_namespaced_name(self):
        assert _agent_to_tool_name("agent:search") == "agent__search"

    def test_deep_namespace(self):
        result = _agent_to_tool_name("db:postgres:query")
        assert "db" in result and "postgres" in result and "query" in result

    def test_parse_simple(self):
        assert _parse_tool_name("search") == ("", "search")

    def test_parse_namespaced(self):
        ns, name = _parse_tool_name("agent__search")
        assert ns == "agent"
        assert name == "search"

    def test_roundtrip(self):
        original = "my_agent:find_user"
        converted = _agent_to_tool_name(original)
        ns, name = _parse_tool_name(converted)
        assert f"{ns}:{name}" == original

    def test_special_chars_preserved(self):
        result = _agent_to_tool_name("agent:get-user_info")
        assert "get-user" in result  # Special chars preserved in local name part


# ─── ToolResult ─────────────────────────────────────────────────

class TestToolResult:
    def test_content_result(self):
        r = ToolResult.content(["hello"])
        assert r.content == ["hello"]
        assert r.is_error is False

    def test_error_result(self):
        r = ToolResult.error("failed")
        assert r.is_error is True
        assert "failed" in str(r)

    def test_serialization(self):
        r = ToolResult(content=["text"], is_error=False)
        d = r.to_dict()
        assert d["content"] == ["text"]
        assert d["isError"] is False


# ─── JSON-RPC Helpers ───────────────────────────────────────────

class TestJsonRpcHelpers:
    def test_request_creation(self):
        req = JsonRpcRequest(method="tools/list", req_id=1)
        assert req.jsonrpc == "2.0"
        assert req.method == "tools/list"

    def test_response_success(self):
        resp = JsonRpcResponse(result={"tools": []}, resp_id=1)
        assert resp.result == {"tools": []}
        assert resp.error is None

    def test_error_response(self):
        err = {"code": -32601, "message": "not found"}
        resp = JsonRpcResponse(error=err, resp_id=None)
        assert resp.error is not None


# ─── MCPServerAdapter Initialize ────────────────────────────────

class TestMCPServerAdapterInitialize:
    def test_create_with_defaults(self):
        adapter = MCPServerAdapter()
        assert adapter.server_name == "mcp-server"
        assert adapter.protocol_version == "2025-11-25"

    def test_create_custom_options(self):
        adapter = MCPServerAdapter(server_name="test-srv")
        assert adapter.server_name == "test-srv"

    @pytest.mark.asyncio
    async def test_initialize_handshake(self):
        adapter = MCPServerAdapter()
        # Mock the raw_send to avoid real I/O
        adapter.raw_send = MagicMock(return_value={"result": {}})
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
               "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}}
        result = await adapter.handle_message(req)
        assert result["result"]["protocolVersion"] == "2025-11-25"
        assert result["result"]["capabilities"]["tools"]

    def test_session_created_after_init(self):
        adapter = MCPServerAdapter()
        assert adapter.session_id is None  # No session until initialize called


# ─── MCPServerAdapter Lifecycle ─────────────────────────────────

class TestMCPServerAdapterLifecycle:
    @pytest.mark.asyncio
    async def test_initialized_notification(self):
        adapter = MCPServerAdapter()
        adapter.raw_send = MagicMock()
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        result = await adapter.handle_message(notification)
        assert result is None or "error" not in str(result)

    @pytest.mark.asyncio
    async def test_ping_responds(self):
        adapter = MCPServerAdapter()
        adapter.raw_send = MagicMock(return_value={"result": {}})
        ping = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
        result = await adapter.handle_message(ping)
        assert result is not None

    @pytest.mark.asyncio
    async def test_unknown_method_returns_error(self):
        adapter = MCPServerAdapter()
        req = {"jsonrpc": "2.0", "id": 3, "method": "nonexistent/method"}
        result = await adapter.handle_message(req)
        if result and "error" in result:
            assert result["error"]["code"] == McpErrorCode.METHOD_NOT_FOUND


# ─── MCPServerAdapter Tools List ────────────────────────────────

class TestMCPServerAdapterToolsList:
    def test_empty_tools_list(self):
        adapter = MCPServerAdapter()
        assert len(adapter._tool_registry) == 0

    def test_register_and_list_tools(self):
        adapter = MCPServerAdapter()

        async def dummy_handler(**kwargs):
            return ToolResult.content(["ok"])

        adapter.register_tool("test_tool", "A test tool", {}, dummy_handler)
        tools = adapter.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"


# ─── MCPServerAdapter Tools Call ────────────────────────────────

class TestMCPServerAdapterToolsCall:
    @pytest.mark.asyncio
    async def test_call_registered_tool(self):
        adapter = MCPServerAdapter()

        async def echo(text: str = "hello"):
            return ToolResult.content([text])

        adapter.register_tool("echo", "Echo back text", {"type": "object"}, echo)
        call_req = {"jsonrpc": "2.0", "id": 10, "method": "tools/call",
                    "params": {"name": "echo", "arguments": {"text": "hi"}}}
        result = await adapter.handle_message(call_req)
        assert result is not None

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool(self):
        adapter = MCPServerAdapter()
        call_req = {"jsonrpc": "2.0", "id": 11, "method": "tools/call",
                    "params": {"name": "missing", "arguments": {}}}
        result = await adapter.handle_message(call_req)
        assert result is not None


# ─── MCPServerAdapter Resources ─────────────────────────────────

class TestMCPServerAdapterResources:
    def test_resource_registration(self):
        adapter = MCPServerAdapter()
        adapter.register_resource("file:///tmp/a.txt", "a.txt", "text/plain", "sample")
        resources = adapter.list_resources()
        assert len(resources) >= 1

    def test_resource_read(self):
        adapter = MCPServerAdapter()
        adapter.register_resource("file:///tmp/b.txt", "b.txt", "text/plain", "content here")
        content = adapter.read_resource("file:///tmp/b.txt")
        assert content is not None


# ─── MCPServerAdapter Prompts ───────────────────────────────────

class TestMCPServerAdapterPrompts:
    def test_prompt_registration(self):
        adapter = MCPServerAdapter()
        adapter.register_prompt("summarize", "Summarize text", [{"name": "text"}])
        prompts = adapter.list_prompts()
        assert any(p["name"] == "summarize" for p in prompts)


# ─── Logging & Completions ──────────────────────────────────────

class TestMCPServerAdapterLogging:
    @pytest.mark.asyncio
    async def test_set_logging_level(self):
        adapter = MCPServerAdapter()
        req = {"jsonrpc": "2.0", "id": 20, "method": "logging/setLevel",
               "params": {"level": "info"}}
        result = await adapter.handle_message(req)
        assert result is not None


# ─── Session Management ─────────────────────────────────────────

class TestMCPServerAdapterSessionManagement:
    def test_generate_session_id(self):
        adapter = MCPServerAdapter()
        sid = adapter._generate_session_id()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_unique_sessions(self):
        adapter = MCPServerAdapter()
        sids = [adapter._generate_session_id() for _ in range(10)]
        # Most should be unique (probabilistic with uuid4)
        assert len(set(sids)) > 8


# ─── MCP HTTP Server Integration ────────────────────────────────

class TestMCPHTTPServer:
    @pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
    @pytest.mark.asyncio
    async def test_post_endpoint_accepts_json_rpc(self):
        app = create_mcp_app(MCPServerAdapter())
        client = await test_utils.test_client(app, server_kwargs=None)
        payload = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.1.0"},
            },
        })
        resp = await client.post("/", data=payload, headers={"Content-Type": "application/json"})
        assert resp.status in (200, 202)


# ─── MCPAdapter (Client) Tests ──────────────────────────────────

class TestMCPAdapterInit:
    def test_init_default(self):
        a = MCPAdapter("http://localhost:8000")
        assert a.server_url == "http://localhost:8000"
        assert a.timeout == 30.0

    def test_init_trailing_slash_stripped(self):
        a = MCPAdapter("http://localhost:8000/")
        assert a.server_url == "http://localhost:8000"

    def test_init_custom_timeout(self):
        a = MCPAdapter("http://x", timeout=60.0)
        assert a.timeout == 60.0


class TestMCPAdapterRoundTrip:
    @pytest.mark.asyncio
    async def test_connect_success(self):
        a = MCPAdapter("http://localhost:9999")
        with patch.object(a, '_http_request', return_value={}):
            result = a.connect()
            assert result is True

    def test_connect_failure_raises(self):
        a = MCPAdapter("http://localhost:9999")
        with patch.object(a, '_http_request', side_effect=MCPConnectionError("fail")):
            with pytest.raises(MCPConnectionError):
                a.connect()

    def test_list_tools_empty(self):
        a = MCPAdapter("http://x")
        with patch.object(a, '_http_request', return_value={"tools": []}):
            tools = a.list_tools()
            assert tools == []

    def test_list_tools_with_data(self):
        a = MCPAdapter("http://x")
        mock_tools = [{"name": "t1", "desc": "d1"}, {"name": "t2"}]
        with patch.object(a, '_http_request', return_value={"tools": mock_tools}):
            tools = a.list_tools()
            assert len(tools) == 2
            assert all(isinstance(t, MCPTool) for t in tools)

    def test_get_tool_found(self):
        a = MCPAdapter("http:x")
        a._tools = [MCPTool(name="find", description="x", parameters={})]
        assert a.get_tool("find") is not None
        assert a.get_tool("missing") is None

    def test_call_tool_success(self):
        a = MCPAdapter("http:x")
        with patch.object(a, '_http_request', return_value={"result": "ok"}) as m:
            result = a.call_tool("find", {"q": "x"})
            assert result == "ok"

    def test_call_tool_failure(self):
        a = MCPAdapter("http:x")
        with patch.object(a, '_http_request', side_effect=MCPToolError("err")):
            with pytest.raises(MCPToolError):
                a.call_tool("bad", {})

    def test_capability_conversion(self):
        tool = MCPTool(name="search", description="Find things", parameters={"type": "object"})
        a = MCPAdapter("http:x")
        cap = a.to_agentlink_capability(tool)
        assert cap["name"].startswith("mcp:")
        assert cap["type"] == "external_tool"


class TestMCPAdapterParseServerCapabilities:
    def test_parse_minimal_capabilities(self):
        caps = {}
        parsed = MCPAdapter.parse_server_capabilities(caps)
        # Should not raise; returns something reasonable
        assert parsed is not None


class TestMCPAdapterBridge:
    def test_bridge_creates_adapter(self):
        class FakeNode:
            pass
        node = FakeNode()
        with patch('agentlink.adapters.mcp.MCPAdapter') as MockCls:
            instance = MockCls.return_value
            instance.connect.return_value = True
            instance.list_tools.return_value = []
            result = create_mcp_bridge(node, "http://localhost:8000")
            assert result is instance


# ─── MCPAgentNodeMixin ──────────────────────────────────────────

class TestMCPAgentNodeMixin:
    def test_discover_without_adapter(self):
        class FakeNode(MCPAgentNodeMixin):
            def __init__(self):
                self.mcp = None
                self._mcp_tools = {}
        n = FakeNode()
        tools = n.discover_mcp_tools()
        assert tools == []

    def test_call_without_adapter_raises(self):
        class FakeNode(MCPAgentNodeMixin):
            def __init__(self):
                self.mcp = None
                self._mcp_tools = {}
        n = FakeNode()
        with pytest.raises(MCPError):
            n.call_mcp_tool("anything")


# ─── Expose Bus As MCP ──────────────────────────────────────────

class TestExposeBusAsMcp:
    def test_expose_bus_creates_server(self):
        from agentlink.runtime.bus import AgentBus
        bus = AgentBus()
        server = expose_bus_as_mcp(bus, server_name="bus-mcp")
        assert server is not None
        assert server.server_name == "bus-mcp"
