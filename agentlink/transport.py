"""
WebSocket transport for remote agent communication (Issue #4).

For distributed deployments (agents on different machines/pods), a persistent
WebSocket connection is more efficient than repeated HTTP requests. This module
provides a ``WSTransport`` that serializes AgentMessage objects over WebSockets,
with server and client modes, plus a ``WSBridge`` that connects a transport to
an AgentBus.

Requires: ``pip install websockets``

Usage (server)::

    from agentlink.transport import WSTransport
    transport = WSTransport(port=8765)
    await transport.start()          # listens for connections

Usage (client)::

    client = WSTransport()
    await client.connect("ws://orchestrator:8765")
    await client.send(message)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, Optional

from agentlink.protocol.message import AgentMessage, AgentAddress, MessageType


def serialize_message(message: AgentMessage) -> Dict[str, Any]:
    """Serialize an AgentMessage to a JSON-ready dict."""
    return message.to_dict()


def deserialize_message(data: Dict[str, Any]) -> AgentMessage:
    """Reconstruct an AgentMessage from a dict."""
    return AgentMessage.from_dict(data)


class WSTransport:
    """
    WebSocket transport for AgentMessage objects.

    Supports server mode (``start``) and client mode (``connect``). Received
    messages are dispatched to the ``on_message`` callback (if set).
    """

    def __init__(
        self,
        url: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8765,
        on_message: Optional[Callable[[AgentMessage], Any]] = None,
    ) -> None:
        self.url = url
        self.host = host
        self.port = port
        self.on_message = on_message
        self._connection = None
        self._server = None

    # ── Client ─────────────────────────────────────────────────────────────

    async def connect(self, url: Optional[str] = None) -> "WSTransport":
        """Connect to a remote WebSocket server (client mode)."""
        import websockets

        self.url = url or self.url
        if not self.url:
            raise ValueError("No WebSocket URL provided to connect()")
        self._connection = await websockets.connect(self.url)
        return self

    # ── Server ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start a WebSocket server, listening on host:port."""
        import websockets

        self._server = await websockets.serve(self._handle, self.host, self.port)

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def _handle(self, websocket) -> None:
        async for raw in websocket:
            data = json.loads(raw)
            message = deserialize_message(data)
            if self.on_message:
                result = self.on_message(message)
                if result is not None:
                    if isinstance(result, AgentMessage):
                        await websocket.send(json.dumps(serialize_message(result)))
                    else:
                        await websocket.send(json.dumps(result))

    # ── Send / receive ─────────────────────────────────────────────────────

    async def send(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Send a message over the WebSocket connection (client mode).

        Returns the reply message, if any.
        """
        if self._connection is None:
            raise RuntimeError("Not connected. Call connect() first.")
        await self._connection.send(json.dumps(serialize_message(message)))
        reply_raw = await self._connection.recv()
        if reply_raw:
            return deserialize_message(json.loads(reply_raw))
        return None

    # ── Convenience ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return f"WSTransport(url={self.url!r}, host={self.host!r}, port={self.port})"


class WSBridge:
    """
    Connect a WSTransport to an AgentBus so remote messages are routed locally.

    Incoming messages (from the WebSocket) are routed through the bus; replies
    are sent back over the transport.
    """

    def __init__(self, transport: WSTransport, bus: Any) -> None:
        self.transport = transport
        self.bus = bus
        transport.on_message = self._on_message

    async def _on_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        try:
            reply = self.bus._route(message, timeout=30.0)
            return reply.to_dict() if reply else None
        except Exception as e:  # noqa: BLE001
            error = AgentMessage(
                type=MessageType.ERROR,
                sender=message.recipient,
                recipient=message.sender,
                content={"error": str(e)},
                correlation_id=message.id,
            )
            return error.to_dict()

    def __repr__(self) -> str:  # pragma: no cover
        return f"WSBridge(transport={self.transport!r})"
