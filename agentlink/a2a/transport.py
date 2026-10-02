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
  * streaming (v1.0 "Send Streaming Message"): :meth:`A2ATransport.stream` /
    :meth:`A2ATransport.astream` POST with ``Accept: text/event-stream`` and
    decode the SSE event stream into reply messages;
    :meth:`A2ATransport.parse_stream_request` decodes inbound streaming
    requests; :meth:`A2ATransport.build_stream_response` produces the full
    SSE body (Task open → status updates → artifact chunks → final update),
    shaped per the v1.0 task-lifecycle stream pattern.

Zero external dependencies — HTTP uses the standard library (``urllib``).
Tolerant on the receiving side: accepts both the v1.0 method names
(``SendMessage`` / ``SendStreamingMessage``) and the legacy v0.2/v0.3
spellings (``message/send`` / ``message/stream``).
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union

from agentlink.a2a import mapping
from agentlink.a2a.card import AgentCard
from agentlink.protocol.message import AgentAddress, AgentMessage, MessageType

# Method names accepted on the inbound side (v1.0 + legacy spellings).
_SEND_METHODS = {"sendmessage", "message/send", "message:send"}
# Method names accepted for streaming (v1.0 + legacy spellings).
_STREAM_METHODS = {"sendstreamingmessage", "message/stream", "message:stream"}

# v1.0 JSON-RPC method names emitted on the wire.
STREAM_METHOD = "SendStreamingMessage"


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

    # ── Streaming (v1.0 "Send Streaming Message", SSE binding) ──────────────

    def build_stream_request(
        self, message: AgentMessage, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build the JSON-RPC ``SendStreamingMessage`` request for a message."""
        request = self.build_request(message, request_id=request_id)
        request["method"] = STREAM_METHOD
        return request

    def parse_stream_events(
        self,
        body: Union[str, bytes],
        sent: Optional[AgentMessage] = None,
    ) -> List[AgentMessage]:
        """
        Parse an SSE body (``text/event-stream``) into reply messages.

        Each ``data:`` block carries one JSON-RPC response whose ``result``
        is a streamed event (Task, ``status-update``, ``artifact-update``,
        or a direct Message). Events without a content payload (pure status
        changes) are skipped; SSE comments / keep-alives are ignored.

        ``sent`` (optional) correlates replies back to the outgoing message
        (reply type + correlation id), mirroring :meth:`parse_response`.

        Raises:
            A2ATransportError: On a JSON-RPC error event, or when the body is
                a plain JSON error response (e.g. the server does not
                support streaming).
        """
        if isinstance(body, bytes):
            body = body.decode("utf-8")

        # Non-SSE error body: a plain JSON-RPC error response.
        if "data:" not in body:
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                raise A2ATransportError(
                    f"A2A stream body is neither SSE nor JSON: {body[:200]!r}"
                ) from None
            if isinstance(data, dict) and "error" in data:
                err = data.get("error") or {}
                raise A2ATransportError(
                    f"A2A stream error {err.get('code')}: {err.get('message')}"
                )
            raise A2ATransportError("A2A stream response carries no events")

        replies: List[AgentMessage] = []
        for block in re.split(r"\r?\n\r?\n", body):
            data_lines = [
                line[5:].strip() for line in block.splitlines()
                if line.startswith("data:")
            ]
            if not data_lines:
                continue  # comment / keep-alive / unknown field lines
            payload = json.loads("\n".join(data_lines))
            if "error" in payload:
                err = payload.get("error") or {}
                raise A2ATransportError(
                    f"A2A stream error {err.get('code')}: {err.get('message')}"
                )
            result = payload.get("result")
            if not isinstance(result, dict):
                continue
            msg = mapping.stream_event_to_agent_message(result)
            if msg is not None:
                if sent is not None:
                    if msg.type is not MessageType.REPLY:
                        msg.type = MessageType.REPLY
                    if not msg.correlation_id:
                        msg.correlation_id = sent.id
                replies.append(msg)
        return replies

    def stream(self, message: AgentMessage) -> List[AgentMessage]:
        """
        Send a streaming request and collect the SSE event stream.

        Returns the decoded reply messages — one per artifact chunk and per
        status update that carries a message payload, in arrival order.
        """
        if not self.url:
            raise A2ATransportError("A2ATransport has no url configured")
        request = self.build_stream_request(message)
        body = json.dumps(request).encode("utf-8")
        try:
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                self.url,
                data=body,
                headers={
                    **{
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                    **self.headers,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to reach A2A agent at {self.url}: {e}")

        return self.parse_stream_events(raw, sent=message)

    async def astream(self, message: AgentMessage) -> List[AgentMessage]:
        """Async variant of :meth:`stream` (runs the blocking call in a thread)."""
        return await asyncio.to_thread(self.stream, message)

    # ── Server side ─────────────────────────────────────────────────────────

    def parse_request(self, request: Dict[str, Any]) -> AgentMessage:
        """
        Decode an inbound JSON-RPC request into an AgentMessage (lossless when
        the client embeds the AgentLink envelope).

        Accepts ``SendMessage`` (v1.0) and ``message/send`` (legacy).
        """
        self._check_method(request, _SEND_METHODS)
        return self._decode_message_params(request)

    def parse_stream_request(self, request: Dict[str, Any]) -> AgentMessage:
        """
        Decode an inbound streaming request (v1.0 ``SendStreamingMessage`` or
        legacy ``message/stream``) into the initiating AgentMessage.

        The reply is delivered by :meth:`build_stream_response`.
        """
        self._check_method(request, _STREAM_METHODS)
        return self._decode_message_params(request)

    @staticmethod
    def _check_method(request: Dict[str, Any], allowed: "frozenset[str] | set[str]") -> None:
        method = str(request.get("method", "")).strip()
        if method.lower() not in allowed:
            raise A2ATransportError(f"Unsupported A2A method: {method!r}")

    def _decode_message_params(self, request: Dict[str, Any]) -> AgentMessage:
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

    # ── Server-side streaming ───────────────────────────────────────────────

    @staticmethod
    def build_stream_events(
        chunks: Sequence[AgentMessage],
        request_id: Union[str, int, None] = None,
        *,
        task_id: Optional[str] = None,
        context_id: Optional[str] = None,
        final_state: str = mapping.TASK_STATE_COMPLETED,
    ) -> List[Dict[str, Any]]:
        """
        Build the JSON-RPC event sequence for a streamed reply, following the
        A2A v1.0 *task-lifecycle stream* pattern:

            1. the initial ``Task`` object (``kind: "task"``),
            2. a ``status-update`` (state ``WORKING``) before every chunk
               (pure state events, no payload — content travels only via
               artifacts so nothing is decoded twice),
            3. one ``artifact-update`` per chunk (content payload; the final
               chunk carries ``lastChunk: true``),
            4. the terminal ``status-update`` (``final: true``).

        With no chunks at all the pattern collapses to Task → final
        status-update.
        """
        chunks = list(chunks)
        tid = task_id or (chunks[0].id if chunks else str(uuid.uuid4()))
        cid = context_id or (
            chunks[0].correlation_id if chunks and chunks[0].correlation_id else str(uuid.uuid4())
        )
        sentinel = AgentMessage(
            type=MessageType.REQUEST,
            sender=AgentAddress("a2a-agent", "a2a"),
            recipient=AgentAddress("a2a-client", "a2a"),
            content="",
            id=tid,
            correlation_id=cid,
        )

        events: List[Dict[str, Any]] = [
            {"jsonrpc": "2.0", "id": request_id, "result": mapping.task_object(sentinel)}
        ]
        for i, chunk in enumerate(chunks):
            more = i < len(chunks) - 1
            events.append({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": mapping.status_update_event(
                    None, mapping.TASK_STATE_WORKING, task_id=tid, context_id=cid,
                ),
            })
            events.append({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": mapping.artifact_update_event(
                    chunk.content,
                    task_id=tid,
                    context_id=cid,
                    metadata=dict(chunk.metadata or {}),
                    last_chunk=not more,
                ),
            })
        events.append({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": mapping.status_update_event(
                None, final_state, task_id=tid, context_id=cid, final=True,
            ),
        })
        return events

    @staticmethod
    def build_sse_body(events: Sequence[Dict[str, Any]]) -> str:
        """
        Serialize JSON-RPC events into an SSE body (``data: <json>`` blocks
        separated by blank lines, the shape A2A's SSE binding prescribes).
        """
        lines: List[str] = []
        for event in events:
            lines.append(f"data: {json.dumps(event, ensure_ascii=False)}")
            lines.append("")
            lines.append("")
        return "\n".join(lines)

    def build_stream_response(
        self,
        chunks: Sequence[AgentMessage],
        request_id: Union[str, int, None] = None,
        *,
        task_id: Optional[str] = None,
        context_id: Optional[str] = None,
        final_state: str = mapping.TASK_STATE_COMPLETED,
    ) -> str:
        """
        Build the complete SSE response body for a streamed reply.

        Pair with :meth:`parse_stream_request` and serve with content type
        ``text/event-stream``.
        """
        events = self.build_stream_events(
            chunks, request_id, task_id=task_id, context_id=context_id,
            final_state=final_state,
        )
        return self.build_sse_body(events)

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
