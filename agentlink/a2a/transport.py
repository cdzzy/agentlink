"""
A2ATransport — JSON-RPC 2.0 wire transport for the A2A v1.0 compatibility layer.

Bridges the AgentLink :class:`~agentlink.protocol.message.AgentMessage`
envelope to A2A v1.0 JSON-RPC operations:

  * client side: :meth:`A2ATransport.send` / :meth:`A2ATransport.asend` POST a
    JSON-RPC ``SendMessage`` request and decode the returned ``Task`` (or
    direct ``Message``) back into a reply AgentMessage.
  * server side: :meth:`A2ATransport.parse_request` decodes an inbound JSON-RPC
    request into an AgentMessage (lossless via the embedded envelope);
    :meth:`A2ATransport.build_success_response` / ``build_error_response``
    produce the JSON-RPC response (the reply is exposed as a completed ``Task``
    with a native ``Artifact``).

Zero external dependencies — HTTP uses the standard library (``urllib``).
Tolerant on the receiving side: accepts both the v1.0 method name
(``SendMessage``) and the legacy v0.2/v0.3 name (``message/send``).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, Optional, Union

from agentlink.a2a import mapping
from agentlink.a2a.card import AgentCard
from agentlink.protocol.message import AgentAddress, AgentMessage, MessageType

# Method names accepted on the inbound side (v1.0 + legacy spellings).
_SEND_METHODS = {"sendmessage", "message/send", "message:send"}


class A2ATransportError(RuntimeError):
    """Raised when an A2A request/response cannot be processed."""


class A2ATransport:
    """
    A2A v1.0 JSON-RPC transport for AgentMessage envelopes.

    Example (client):
        transport = A2ATransport(url="https://agent.example.com/a2a")
        reply = transport.send(AgentMessage(
            type=MessageType.REQUEST,
            sender=AgentAddress.local("me"),
            recipient=AgentAddress("remote-agent", "a2a"),
            content="Analyze Q1 data",
        ))

    Example (server loop, in-process):
        request = transport.build_request(message)
        inbound = transport.parse_request(request)          # AgentMessage
        reply = inbound.reply("42")
        response = transport.build_success_response(reply, request_id=request["id"])
        got = transport.parse_response(response["result"], inbound)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        method: str = "SendMessage",
        local_agent_id: str = "agentlink",
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.url = url.rstrip("/") if url else None
        self.method = method
        self.local_agent_id = local_agent_id
        self.timeout = timeout
        self.headers = headers or {}

    # ── Discovery ───────────────────────────────────────────────────────────

    @classmethod
    def from_card(cls, card: AgentCard, **kwargs: Any) -> "A2ATransport":
        """Build a transport targeting the card's first JSONRPC interface."""
        for interface in card.supportedInterfaces:
            if interface.protocolBinding.upper() == "JSONRPC":
                return cls(url=interface.url, method="SendMessage", **kwargs)
        raise A2ATransportError("Agent Card exposes no JSONRPC interface")

    # ── Client side ─────────────────────────────────────────────────────────

    def build_request(
        self, message: AgentMessage, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build the JSON-RPC 2.0 ``SendMessage`` request for a message."""
        return {
            "jsonrpc": "2.0",
            "id": request_id or str(uuid.uuid4()),
            "method": self.method,
            "params": {
                "message": mapping.agent_message_to_a2a_message(message),
                "metadata": {},
            },
        }

    def send(self, message: AgentMessage) -> AgentMessage:
        """
        Send a message to the remote A2A agent and return the decoded reply.

        Raises:
            A2ATransportError: On transport-level or A2A error responses.
        """
        if not self.url:
            raise A2ATransportError("A2ATransport has no url configured")
        request = self.build_request(message)
        body = json.dumps(request).encode("utf-8")
        try:
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                self.url,
                data=body,
                headers={
                    **{"Content-Type": "application/json", "Accept": "application/json"},
                    **self.headers,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to reach A2A agent at {self.url}: {e}")

        if "error" in data:
            err = data["error"] or {}
            raise A2ATransportError(f"A2A error {err.get('code')}: {err.get('message')}")
        result = data.get("result")
        if result is None:
            raise A2ATransportError("A2A response carries no result")
        return self.parse_response(result, message)

    async def asend(self, message: AgentMessage) -> AgentMessage:
        """Async variant of :meth:`send` (runs the blocking call in a thread)."""
        return await asyncio.to_thread(self.send, message)

    # ── Server side ─────────────────────────────────────────────────────────

    def parse_request(self, request: Dict[str, Any]) -> AgentMessage:
        """
        Decode an inbound JSON-RPC request into an AgentMessage (lossless when
        the client embeds the AgentLink envelope).

        Accepts ``SendMessage`` (v1.0) and ``message/send`` (legacy).
        """
        method = str(request.get("method", "")).strip()
        if method.lower() not in _SEND_METHODS:
            raise A2ATransportError(f"Unsupported A2A method: {method!r}")

        params = request.get("params") or {}
        a2a_message = params.get("message")
        if not isinstance(a2a_message, dict):
            raise A2ATransportError("A2A request params carry no 'message' object")

        default_sender = AgentAddress("a2a-client", "a2a")
        default_recipient = AgentAddress(self.local_agent_id, "a2a")
        return mapping.a2a_message_to_agent_message(
            a2a_message,
            default_sender=default_sender,
            default_recipient=default_recipient,
        )

    def build_success_response(
        self,
        reply: AgentMessage,
        request_id: Union[str, int, None] = None,
        *,
        original: Optional[AgentMessage] = None,
        artifact_name: str = "result",
    ) -> Dict[str, Any]:
        """
        Build a JSON-RPC success response wrapping ``reply`` as a completed
        A2A ``Task`` with a native ``Artifact``.
        """
        task = mapping.agent_message_to_a2a_task(reply, state=mapping.TASK_STATE_COMPLETED)
        task["artifacts"] = [{
            "artifactId": str(uuid.uuid4()),
            "name": artifact_name,
            "parts": [mapping.content_to_part(reply.content)],
            "metadata": {},
        }]
        if original is not None:
            task["id"] = original.id
            task["contextId"] = original.correlation_id or original.id
        return {"jsonrpc": "2.0", "id": request_id, "result": task}

    @staticmethod
    def build_error_response(
        code: int,
        message: str,
        request_id: Union[str, int, None] = None,
    ) -> Dict[str, Any]:
        """Build a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    # ── Response decoding (client side) ─────────────────────────────────────

    def parse_response(
        self,
        result: Dict[str, Any],
        sent: Optional[AgentMessage] = None,
    ) -> AgentMessage:
        """
        Decode an A2A ``SendMessage`` result (a ``Task`` or a direct
        ``Message``) into a reply AgentMessage, preferring the lossless
        AgentLink envelope when present.
        """
        if not isinstance(result, dict):
            raise A2ATransportError("A2A result must be a JSON object")

        if "parts" in result and "status" not in result:      # direct Message
            msg = mapping.a2a_message_to_agent_message(result)
            if sent is not None:
                if msg.type is not MessageType.REPLY:
                    msg.type = MessageType.REPLY
                if not msg.correlation_id:
                    msg.correlation_id = sent.id
            return msg

        return mapping.a2a_task_to_agent_message(result, original=sent)

    def __repr__(self) -> str:  # pragma: no cover
        return f"A2ATransport(url={self.url!r}, method={self.method!r})"
