"""
A2A Protocol Adapter — bridge AgentLink to Google's Agent-to-Agent (A2A) protocol.

A2A (Agent-to-Agent) is Google's open standard for inter-agent communication.
This adapter lets any AgentLink agent act as an A2A server endpoint and/or
connect to external A2A agents.

Reference: https://a2a-protocol.org / https://github.com/google/a2a-python

Usage:
    from agentlink.adapters.a2a_adapter import A2AAdapter, A2AServerAdapter

    # Wrap an AgentLink node as an A2A server
    server = A2AServerAdapter(node=my_node, port=8000)
    await server.start()

    # Or connect to a remote A2A agent
    client = A2AAdapter(agent_url="http://remote-agent:8000")
    reply = await client.send("Analyze Q1 data")
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict

from agentlink.adapters.base import BaseAdapter
from agentlink.protocol.capability import AgentCapability
from agentlink.protocol.message import AgentMessage, MessageType
from agentlink.runtime.node import AgentNode


# ---------------------------------------------------------------------------
# A2A Protocol Types
# ---------------------------------------------------------------------------

@dataclass
class A2ATask:
    """Represents an A2A task/request."""
    id: str
    context: Dict[str, Any]
    status: str = "submitted"  # submitted | working | completed | failed | canceled
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class A2AAgentCard:
    """
    A2A Agent Card — the discovery document for an A2A-compatible agent.

    Published at /.well-known/agent.json per the A2A spec.
    """
    name: str
    description: str
    version: str = "1.0.0"
    capabilities: Dict[str, bool] = field(default_factory=lambda: {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": False,
    })
    skills: List[Dict[str, str]] = field(default_factory=list)
    defaultInputModes: List[str] = field(default_factory=lambda: ["text"])
    defaultOutputModes: List[str] = field(default_factory=lambda: ["text"])
    tags: List[str] = field(default_factory=list)
    provider: Dict[str, str] = field(default_factory=lambda: {
        "organization": "agentconfig",
        "name": "AgentLink",
    })
    url: str = ""
    documentationUrl: str = ""
    endpoint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> A2AAgentCard:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# A2A JSON-RPC Message Helpers
# ---------------------------------------------------------------------------

def a2a_request(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Build an A2A JSON-RPC 2.0 request."""
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params,
    }


def a2a_response(result: Any, request_id: str) -> Dict[str, Any]:
    """Build an A2A JSON-RPC 2.0 success response."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def a2a_error(code: int, message: str, request_id: str) -> Dict[str, Any]:
    """Build an A2A JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


# ---------------------------------------------------------------------------
# A2AAdapter — connect to a remote A2A agent
# ---------------------------------------------------------------------------

class A2AAdapter(BaseAdapter):
    """
    Connect AgentLink to a remote A2A agent via HTTP.

    This adapter lets you call any A2A-compatible agent as if it were
    a local AgentLink node. It handles JSON-RPC 2.0 encoding/decoding.

    Example:
        from agentlink.adapters.a2a_adapter import A2AAdapter

        remote = A2AAdapter(
            agent_url="http://analyst-agent:8000",
            agent_id="remote-analyst",
            capabilities=["data-analysis"],
        )
        node = remote.as_node()
        bus.register(node)
        result = node.send("analyst", "What were Q1 sales?")
    """

    def __init__(
        self,
        agent_url: str,
        agent_id: str = "a2a-remote",
        namespace: str = "a2a",
        capabilities: Optional[List[Union[str, AgentCapability]]] = None,
        description: str = "Remote A2A agent",
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(agent_id, namespace, capabilities, description)
        self.agent_url = agent_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}

    async def _ainvoke(self, message: AgentMessage) -> Any:
        """Async invoke — call the remote A2A agent via JSON-RPC."""
        try:
            import urllib.request
            import urllib.error

            params = {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "content": message.content,
                    "metadata": message.metadata or {},
                },
            }
            body = json.dumps(a2a_request("agent/sendMessage", params)).encode()
            req = urllib.request.Request(
                f"{self.agent_url}/a2a",
                data=body,
                headers={
                    **{"Content-Type": "application/json", "Accept": "application/json"},
                    **self.headers,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                if "error" in data:
                    raise RuntimeError(f"A2A remote error: {data['error']}")
                result = data.get("result", {})
                return result.get("content", str(result))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to A2A agent at {self.agent_url}: {e}")

    def _invoke(self, message: AgentMessage) -> Any:
        """Sync invoke wrapper."""
        try:
            import urllib.request
            import urllib.error

            params = {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "content": message.content,
                    "metadata": message.metadata or {},
                },
            }
            body = json.dumps(a2a_request("agent/sendMessage", params)).encode()
            req = urllib.request.Request(
                f"{self.agent_url}/a2a",
                data=body,
                headers={
                    **{"Content-Type": "application/json", "Accept": "application/json"},
                    **self.headers,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                if "error" in data:
                    raise RuntimeError(f"A2A remote error: {data['error']}")
                result = data.get("result", {})
                return result.get("content", str(result))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to A2A agent at {self.agent_url}: {e}")

    @classmethod
    async def get_agent_card(cls, url: str, timeout: float = 10.0) -> A2AAgentCard:
        """
        Discover an A2A agent by fetching its Agent Card.

        Per the A2A spec, the agent card is published at /.well-known/agent.json
        """
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(
                f"{url.rstrip('/')}/.well-known/agent.json",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                return A2AAgentCard.from_dict(data)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(f"No A2A Agent Card found at {url}/.well-known/agent.json")
            raise

    @classmethod
    def discover_and_wrap(
        cls,
        agent_url: str,
        timeout: float = 10.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> A2AAdapter:
        """
        Auto-discover a remote A2A agent and wrap it as an AgentLink node.

        Fetches the agent card, then creates an A2AAdapter with
        the discovered metadata.
        """
        card = cls.get_agent_card(agent_url, timeout)
        return cls(
            agent_url=agent_url,
            agent_id=card.name,
            capabilities=[s.get("name", s.get("id", "")) for s in card.skills],
            description=card.description,
            timeout=timeout,
            headers=headers,
        )


# ---------------------------------------------------------------------------
# A2AServerAdapter — expose an AgentLink node as an A2A server
# ---------------------------------------------------------------------------

class A2AServerAdapter:
    """
    Expose an AgentLink AgentNode as an A2A server endpoint.

    This adapter:
    1. Publishes an A2A Agent Card at /.well-known/agent.json
    2. Handles incoming A2A JSON-RPC requests via HTTP POST /a2a
    3. Translates A2A messages to AgentLink messages and calls the node

    The server is a minimal, zero-dependency HTTP implementation.
    For production, deploy behind gunicorn/uwsgi or behind a reverse proxy.

    Example:
        from agentlink.adapters.a2a_adapter import A2AServerAdapter

        server = A2AServerAdapter(
            node=my_node,
            host="0.0.0.0",
            port=8000,
            agent_card=A2AAgentCard(
                name="researcher",
                description="Research agent for market data analysis",
                skills=[{"id": "web-search", "name": "Web Search"}],
            ),
        )
        await server.start()
    """

    def __init__(
        self,
        node: AgentNode,
        host: str = "0.0.0.0",
        port: int = 8000,
        agent_card: Optional[A2AAgentCard] = None,
        max_connections: int = 100,
    ):
        self.node = node
        self.host = host
        self.port = port
        self.max_connections = max_connections

        # Build default agent card from the node
        self.agent_card = agent_card or A2AAgentCard(
            name=node.agent_id,
            description=node.description or f"AgentLink agent: {node.agent_id}",
            skills=[
                {"id": cap if isinstance(cap, str) else cap.name, "name": cap if isinstance(cap, str) else cap.name}
                for cap in (node.capabilities or [])
            ] if node.capabilities else [],
            endpoint=f"http://{host}:{port}",
        )

    async def _handle_jsonrpc(self, body: bytes) -> bytes:
        """Handle an incoming A2A JSON-RPC request."""
        try:
            request = json.loads(body)
        except json.JSONDecodeError:
            return json.dumps(a2a_error(-32700, "Parse error", "")).encode()

        request_id = request.get("id", "")
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "agent/sendMessage":
            return await self._handle_send_message(request_id, params)
        elif method == "agent/getTask":
            task_id = params.get("taskId", "")
            return await self._handle_get_task(request_id, task_id)
        else:
            return json.dumps(a2a_error(-32601, f"Method not found: {method}", request_id)).encode()

    async def _handle_send_message(self, request_id: str, params: Dict[str, Any]) -> bytes:
        """Handle agent/sendMessage — forward to the AgentLink node."""
        msg_data = params.get("message", {})
        content = msg_data.get("content", "")
        role = msg_data.get("role", "user")

        # Convert to AgentLink message
        msg = AgentMessage(
            type=MessageType.REQUEST,
            content=content,
            metadata={"a2a_role": role, "source": "a2a"},
        )

        try:
            reply = self.node.send_sync(msg)
            result = {
                "content": reply.content,
                "metadata": reply.metadata or {},
            }
            return json.dumps(a2a_response(result, request_id)).encode()
        except Exception as e:
            return json.dumps(a2a_error(-32603, f"Internal error: {e}", request_id)).encode()

    async def _handle_get_task(self, request_id: str, task_id: str) -> bytes:
        """Handle agent/getTask — return task status (stub for A2A spec compat)."""
        return json.dumps(a2a_response({"taskId": task_id, "status": "unknown"}, request_id)).encode()

    def get_agent_card_json(self) -> bytes:
        """Return the A2A Agent Card as JSON bytes."""
        card = self.agent_card.to_dict()
        return json.dumps(card, indent=2).encode("utf-8")

    async def start(self) -> None:
        """
        Start the A2A HTTP server.
        Uses asyncio to run a minimal HTTP server without extra dependencies.
        """
        import asyncio
        import os

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            addr = writer.get_extra_info("peername")
            try:
                request_line = await reader.readline()
                if not request_line:
                    return

                parts = request_line.decode().strip().split(" ")
                if len(parts) < 2:
                    writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                    await writer.drain()
                    return

                method, path, *_ = parts

                # Read headers
                headers: Dict[str, str] = {}
                content_length = 0
                while True:
                    line = await reader.readline()
                    if line in (b"\r\n", b"\n", b""):
                        break
                    key, _, value = line.decode().partition(":")
                    headers[key.strip().lower()] = value.strip()
                content_length = int(headers.get("content-length", 0))

                # Route
                if method == "GET" and path == "/.well-known/agent.json":
                    writer.write(
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: application/json\r\n"
                        b"Access-Control-Allow-Origin: *\r\n\r\n"
                    )
                    writer.write(self.get_agent_card_json())
                elif method == "POST" and path == "/a2a":
                    body = await reader.read(content_length)
                    response_body = await self._handle_jsonrpc(body)
                    writer.write(
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: application/json\r\n"
                        f"Content-Length: {len(response_body)}\r\n\r\n".encode()
                    )
                    writer.write(response_body)
                else:
                    writer.write(b"HTTP/1.1 404 Not Found\r\n\r\n")

                await writer.drain()
            except Exception as e:
                print(f"[A2AServerAdapter] Error handling {addr}: {e}")
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        server = await asyncio.start_server(handler, self.host, self.port)
        addr = server.sockets[0].getsockname()
        print(f"[A2AServerAdapter] A2A server listening on http://{addr[0]}:{addr[1]}")
        print(f"[A2AServerAdapter] Agent Card: http://{addr[0]}:{addr[1]}/.well-known/agent.json")

        async with server:
            await server.serve_forever()
