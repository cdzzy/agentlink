"""
Tests for the MCP Streamable HTTP transport (2025-03-26+ spec).

Covers content negotiation (JSON vs SSE), session lifecycle (initialize →
follow-up requests → DELETE), the server-to-client SSE channel, CORS preflight,
and JSON-RPC error paths.
"""

import json

import pytest

# aiohttp is an optional dependency for HTTP transport tests
try:
    from aiohttp import test_utils

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from agentlink.adapters.mcp import (
    MCP_PROTOCOL_HEADER,
    MCP_SESSION_HEADER,
    McpErrorCode,
    MCPServerAdapter,
    ToolResult,
    create_streamable_mcp_app,
)

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


def _make_server() -> MCPServerAdapter:
    server = MCPServerAdapter(server_name="test-agent")

    async def echo(**kwargs):
        return ToolResult.make([{"type": "text", "text": kwargs.get("text", "")}])

    server.register_tool("echo", "Echo text back", {}, echo)
    return server


async def _initialize(client, session_required: bool = False) -> str:
    """Run the initialize handshake and return the minted session id."""
    resp = await client.post(
        "/mcp",
        data=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-11-25", "clientInfo": {"name": "pytest"}},
            }
        ),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    assert resp.status == 200
    return resp.headers[MCP_SESSION_HEADER]


class TestSessionLifecycle:
    async def test_initialize_mints_session_id(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            sid = await _initialize(client)

            assert sid
            payload = await client.post(
                "/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
                headers={"Content-Type": "application/json"},
            )
            # Each initialize mints an independent session
            assert payload.status == 200
            assert payload.headers[MCP_SESSION_HEADER] != sid

    async def test_follow_up_request_carries_session(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            sid = await _initialize(client)
            resp = await client.post(
                "/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                headers={"Content-Type": "application/json", MCP_SESSION_HEADER: sid},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["result"]["tools"][0]["name"] == "echo"

    async def test_unknown_session_is_rejected(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.post(
                "/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                headers={"Content-Type": "application/json", MCP_SESSION_HEADER: "bogus"},
            )
            assert resp.status == 404
            body = await resp.json()
            assert body["error"]["code"] == int(McpErrorCode.INVALID_REQUEST)

    async def test_session_required_rejects_missing_header(self):
        server = _make_server()
        app = create_streamable_mcp_app(server, session_required=True)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.post(
                "/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            body = await resp.json()
            assert "Missing Mcp-Session-Id" in body["error"]["message"]

    async def test_session_not_required_by_default(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.post(
                "/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 200

    async def test_delete_terminates_session(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            sid = await _initialize(client)

            deleted = await client.delete("/mcp", headers={MCP_SESSION_HEADER: sid})
            assert deleted.status == 204

            # The session is gone, so a follow-up is now unknown.
            resp = await client.post(
                "/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
                headers={"Content-Type": "application/json", MCP_SESSION_HEADER: sid},
            )
            assert resp.status == 404

    async def test_delete_without_session_header(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.delete("/mcp")
            assert resp.status == 400


class TestContentNegotiation:
    async def test_json_response_by_default(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            sid = await _initialize(client)
            resp = await client.post(
                "/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                headers={"Content-Type": "application/json", MCP_SESSION_HEADER: sid},
            )
            assert resp.status == 200
            assert resp.content_type == "application/json"
            assert (await resp.json())["jsonrpc"] == "2.0"

    async def test_sse_response_when_client_accepts_event_stream(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            sid = await _initialize(client)
            resp = await client.post(
                "/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    MCP_SESSION_HEADER: sid,
                },
            )
            assert resp.status == 200
            assert resp.content_type == "text/event-stream"
            raw = await resp.text()
            assert "event: message" in raw
            data_line = [ln for ln in raw.splitlines() if ln.startswith("data: ")][0]
            payload = json.loads(data_line[len("data: "):])
            assert payload["jsonrpc"] == "2.0"
            assert payload["result"]["tools"][0]["name"] == "echo"

    async def test_notification_returns_202(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            sid = await _initialize(client)
            resp = await client.post(
                "/mcp",
                # No "id" → a notification, which must not carry a body.
                data=json.dumps(
                    {"jsonrpc": "2.0", "method": "notifications/initialized"}
                ),
                headers={"Content-Type": "application/json", MCP_SESSION_HEADER: sid},
            )
            assert resp.status == 202


class TestSSEChannel:
    async def test_plain_get_returns_server_metadata(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.get("/mcp")
            assert resp.status == 200
            body = await resp.json()
            assert body["name"] == "test-agent"
            assert body["transport"] == "streamable-http"
            assert body["endpoint"] == "/mcp"

    async def test_sse_get_requires_a_session(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.get("/mcp", headers={"Accept": "text/event-stream"})
            assert resp.status == 400

    async def test_sse_get_opens_stream_for_valid_session(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            sid = await _initialize(client)
            resp = await client.get(
                "/mcp",
                headers={"Accept": "text/event-stream", MCP_SESSION_HEADER: sid},
            )
            assert resp.status == 200
            assert resp.content_type == "text/event-stream"
            assert MCP_SESSION_HEADER in resp.headers
            raw = await resp.text()
            assert ": stream-open" in raw


class TestCorsAndErrors:
    async def test_options_preflight(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.options("/mcp")
            assert resp.status == 204
            assert resp.headers["Access-Control-Allow-Origin"] == "*"
            assert "POST" in resp.headers["Access-Control-Allow-Methods"]
            assert MCP_SESSION_HEADER in resp.headers["Access-Control-Allow-Headers"]

    async def test_invalid_json_returns_parse_error(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.post(
                "/mcp",
                data="not-json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            body = await resp.json()
            assert body["error"]["code"] == int(McpErrorCode.PARSE_ERROR)

    async def test_non_object_payload_rejected(self):
        server = _make_server()
        app = create_streamable_mcp_app(server)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.post(
                "/mcp",
                data=json.dumps([1, 2, 3]),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            body = await resp.json()
            assert body["error"]["code"] == int(McpErrorCode.INVALID_REQUEST)

    async def test_custom_path(self):
        server = _make_server()
        app = create_streamable_mcp_app(server, path="/mcp/v1")

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.get("/mcp/v1")
            assert resp.status == 200
            body = await resp.json()
            assert body["endpoint"] == "/mcp/v1"

    async def test_protocol_header_constant(self):
        assert MCP_PROTOCOL_HEADER == "MCP-Protocol-Version"
        assert MCP_SESSION_HEADER == "Mcp-Session-Id"
