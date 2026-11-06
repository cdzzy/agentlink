"""
AG-UI (Agent User Interaction Protocol) adapter for AgentLink.

AG-UI is the third leg of the agent protocol stack: where MCP connects an
agent to **tools** and A2A connects an agent to **other agents**, AG-UI
connects an agent to the **human** in front of it — a streaming, event-based
protocol consumed by frontends (CopilotKit, LangGraph Studio, custom UIs).

This module implements the producer side of AG-UI 1.0:

- **Event emitter** — ``AGUIEmitter`` builds spec-shaped events for a run
  (``RUN_STARTED`` / ``TEXT_MESSAGE_*`` / ``TOOL_CALL_*`` / ``STATE_*`` /
  ``RUN_FINISHED`` ...), tracking message and tool-call ids across events.
- **Human-in-the-loop** — ``AGUIInterruptManager`` implements the AG-UI
  interrupt contract: a run pauses by finishing with
  ``outcome={"type": "interrupt", "interrupts": [...]}``, and the next run
  resumes through ``RunAgentInput["resume"]``. Approval payloads are
  validated against the interrupt's ``responseSchema``.
- **SSE transport** — ``create_ag_ui_app(handler)`` serves ``POST /`` with a
  ``text/event-stream`` response, exactly what an AG-UI client expects.
- **AgentLink bridge** — ``expose_node_as_ag_ui(node)`` turns any
  ``AgentNode`` into an AG-UI handler, streaming its chunks as
  ``TEXT_MESSAGE_CONTENT`` events.

Reference: https://docs.ag-ui.com

Usage (transport)::

    from aiohttp import web
    from agentlink.adapters.ag_ui import create_ag_ui_app, expose_node_as_ag_ui

    app = create_ag_ui_app(expose_node_as_ag_ui(node))
    web.run_app(app, port=8000)      # POST / with a RunAgentInput

Usage (emit by hand)::

    emitter = AGUIEmitter(thread_id="t-1", run_id="r-1")
    yield emitter.run_started()
    msg = emitter.text_message_start()
    yield msg
    yield emitter.text_message_content("Hello", msg["messageId"])
    yield emitter.text_message_end()
    yield emitter.run_finished(result="Hello")

Usage (approval gate)::

    manager = AGUIInterruptManager()
    interrupt = manager.create(
        reason="tool_call",
        message="Send this email?",
        tool_call_id="tc-1",
    )
    yield emitter.run_finished(outcome=manager.outcome())

    # Next run, from RunAgentInput["resume"]:
    manager.resolve({"interruptId": interrupt.id,
                     "status": "resolved",
                     "payload": {"approved": True}})

Only the SSE transport requires aiohttp (``pip install cdzzy-agentlink[agui]``).
The emitter and interrupt manager are dependency-free.
"""

from __future__ import annotations

import inspect
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Iterable, Mapping, Optional

from agentlink.protocol.message import AgentAddress, AgentMessage, MessageType
from agentlink.runtime.stream import is_streamable

AGUI_PROTOCOL_VERSION = "1.0"


# ─── Errors ────────────────────────────────────────────────────────────────────

class AGUIError(Exception):
    """Base error for AG-UI production mistakes."""


class AGUIResumeError(AGUIError):
    """A resume payload is invalid (unknown, expired, or schema mismatch)."""


# ─── Event types ───────────────────────────────────────────────────────────────

class AGUIEventType(str, Enum):
    """The AG-UI 1.0 event catalogue (``@ag-ui/core`` ``EventType``)."""

    # Lifecycle
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"

    # Text messages
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TEXT_MESSAGE_CHUNK = "TEXT_MESSAGE_CHUNK"

    # Tool calls
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"

    # State
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"
    ACTIVITY_SNAPSHOT = "ACTIVITY_SNAPSHOT"
    ACTIVITY_DELTA = "ACTIVITY_DELTA"

    # Reasoning
    REASONING_START = "REASONING_START"
    REASONING_MESSAGE_START = "REASONING_MESSAGE_START"
    REASONING_MESSAGE_CONTENT = "REASONING_MESSAGE_CONTENT"
    REASONING_MESSAGE_END = "REASONING_MESSAGE_END"
    REASONING_MESSAGE_CHUNK = "REASONING_MESSAGE_CHUNK"
    REASONING_END = "REASONING_END"
    REASONING_ENCRYPTED_VALUE = "REASONING_ENCRYPTED_VALUE"

    # Subagents
    SUBAGENT_STARTED = "SUBAGENT_STARTED"
    SUBAGENT_FINISHED = "SUBAGENT_FINISHED"
    SUBAGENT_ERROR = "SUBAGENT_ERROR"

    # Special
    RAW = "RAW"
    CUSTOM = "CUSTOM"


_EVENT_VALUES = frozenset(event.value for event in AGUIEventType)


# ─── Interrupts (human-in-the-loop) ────────────────────────────────────────────

#: Recommended ``responseSchema`` for approval-style interrupts. The presence
#: of ``editedArgs`` is the AG-UI capability signal for approve-with-edits UI.
AGUI_APPROVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "editedArgs": {
            "type": "object",
            "description": "Full replacement of the tool args. Not merged.",
        },
    },
    "required": ["approved"],
}


@dataclass
class AGUIInterrupt:
    """
    A pause in a run that waits for human input.

    Serialised with camelCase keys so it drops straight into a
    ``RUN_FINISHED`` event's ``outcome.interrupts`` array.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reason: str = "confirmation"
    message: Optional[str] = None
    tool_call_id: Optional[str] = None
    response_schema: Optional[dict[str, Any]] = None
    expires_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    subagent_run_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, "reason": self.reason}
        if self.message is not None:
            data["message"] = self.message
        if self.tool_call_id is not None:
            data["toolCallId"] = self.tool_call_id
        if self.response_schema is not None:
            data["responseSchema"] = self.response_schema
        if self.expires_at is not None:
            data["expiresAt"] = self.expires_at
        if self.metadata:
            data["metadata"] = self.metadata
        if self.subagent_run_id is not None:
            data["subagentRunId"] = self.subagent_run_id
        return data


@dataclass
class AGUIResume:
    """One entry of ``RunAgentInput["resume"]``."""

    interrupt_id: str
    status: str = "resolved"
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ("resolved", "cancelled"):
            raise AGUIResumeError(
                f"Resume status must be 'resolved' or 'cancelled', got {self.status!r}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AGUIResume":
        interrupt_id = data.get("interruptId") or data.get("interrupt_id")
        if not interrupt_id:
            raise AGUIResumeError("Resume entry is missing 'interruptId'")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise AGUIResumeError("Resume 'metadata' must be an object")
        return cls(
            interrupt_id=str(interrupt_id),
            status=str(data.get("status", "resolved")),
            payload=data.get("payload"),
            metadata=dict(metadata),
        )

    @classmethod
    def parse_many(cls, items: Iterable[Mapping[str, Any]]) -> list["AGUIResume"]:
        return [cls.from_dict(item) for item in items]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "interruptId": self.interrupt_id,
            "status": self.status,
        }
        if self.payload is not None:
            data["payload"] = self.payload
        if self.metadata:
            data["metadata"] = self.metadata
        return data


class AGUIInterruptManager:
    """
    Tracks open interrupts for a thread and validates incoming resumes.

    Enforces the AG-UI contract rules that matter to a producer: unknown
    interrupt ids, expired interrupts, and payloads that violate the
    declared ``responseSchema`` all raise :class:`AGUIResumeError`.
    """

    def __init__(self) -> None:
        self._open: dict[str, AGUIInterrupt] = {}

    def create(
        self,
        reason: str = "confirmation",
        message: Optional[str] = None,
        *,
        tool_call_id: Optional[str] = None,
        response_schema: Optional[dict[str, Any]] = None,
        expires_at: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        subagent_run_id: Optional[str] = None,
    ) -> AGUIInterrupt:
        """Create and register an interrupt."""
        interrupt = AGUIInterrupt(
            reason=reason,
            message=message,
            tool_call_id=tool_call_id,
            response_schema=response_schema,
            expires_at=expires_at,
            metadata=metadata or {},
            subagent_run_id=subagent_run_id,
        )
        self._open[interrupt.id] = interrupt
        return interrupt

    def get(self, interrupt_id: str) -> Optional[AGUIInterrupt]:
        return self._open.get(interrupt_id)

    @property
    def open(self) -> list[AGUIInterrupt]:
        return list(self._open.values())

    def __len__(self) -> int:
        return len(self._open)

    def outcome(self) -> Optional[dict[str, Any]]:
        """The ``RUN_FINISHED.outcome`` for the currently open interrupts."""
        if not self._open:
            return None
        return {
            "type": "interrupt",
            "interrupts": [interrupt.to_dict() for interrupt in self._open.values()],
        }

    def resolve(
        self,
        resume: AGUIResume | Mapping[str, Any],
        *,
        now: Optional[datetime] = None,
    ) -> AGUIInterrupt:
        """
        Apply a resume entry, returning the interrupt it resolved.

        Raises:
            AGUIResumeError: unknown id, expired interrupt, bad status, or a
                payload that fails the interrupt's ``responseSchema``.
        """
        if isinstance(resume, Mapping):
            resume = AGUIResume.from_dict(resume)

        interrupt = self._open.get(resume.interrupt_id)
        if interrupt is None:
            raise AGUIResumeError(
                f"Unknown or already-resolved interrupt {resume.interrupt_id!r}"
            )

        if interrupt.expires_at:
            expiry = _parse_datetime(interrupt.expires_at)
            current = now or datetime.now(timezone.utc)
            if expiry is not None and current >= expiry:
                self._open.pop(interrupt.id, None)
                raise AGUIResumeError(
                    f"Interrupt {interrupt.id!r} expired at {interrupt.expires_at}"
                )

        if resume.status == "resolved" and interrupt.response_schema is not None:
            _validate_payload(interrupt.response_schema, resume.payload)

        self._open.pop(interrupt.id, None)
        return interrupt


# ─── Event emitter ─────────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(time.time() * 1000)


class AGUIEmitter:
    """
    Builds AG-UI events for one run, tracking ids across the lifecycle.

    Every method returns a plain ``dict`` ready to be JSON-serialised and
    written as an SSE ``data:`` frame. ``None`` is returned for events that
    the spec forbids — currently empty ``TEXT_MESSAGE_CONTENT`` deltas —
    so producers can filter falsy values.
    """

    def __init__(
        self,
        thread_id: str,
        run_id: Optional[str] = None,
        *,
        parent_run_id: Optional[str] = None,
    ) -> None:
        if not thread_id:
            raise AGUIError("AGUIEmitter requires a thread_id")
        self.thread_id = thread_id
        self.run_id = run_id or str(uuid.uuid4())
        self.parent_run_id = parent_run_id
        self._text_message_id: Optional[str] = None
        self._open_tool_calls: set[str] = set()

    # ── Internal ──────────────────────────────────────────────

    def _emit(self, event_type: AGUIEventType, **payload: Any) -> dict[str, Any]:
        event: dict[str, Any] = {"type": event_type.value, "timestamp": _now_ms()}
        event.update({key: value for key, value in payload.items() if value is not None})
        return event

    # ── Introspection ─────────────────────────────────────────

    @property
    def current_message_id(self) -> Optional[str]:
        """The text message currently being streamed, if any."""
        return self._text_message_id

    # ── Lifecycle ─────────────────────────────────────────────

    def run_started(self, input: Optional[Any] = None) -> dict[str, Any]:
        return self._emit(
            AGUIEventType.RUN_STARTED,
            threadId=self.thread_id,
            runId=self.run_id,
            parentRunId=self.parent_run_id,
            input=input,
        )

    def run_finished(
        self,
        result: Any = None,
        *,
        outcome: Optional[dict[str, Any]] = None,
        usage: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        return self._emit(
            AGUIEventType.RUN_FINISHED,
            threadId=self.thread_id,
            runId=self.run_id,
            result=result,
            outcome=outcome,
            usage=usage,
        )

    def run_error(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        usage: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        return self._emit(
            AGUIEventType.RUN_ERROR,
            message=message,
            code=code,
            usage=usage,
        )

    def step_started(self, step_name: str) -> dict[str, Any]:
        return self._emit(AGUIEventType.STEP_STARTED, stepName=step_name)

    def step_finished(self, step_name: str) -> dict[str, Any]:
        return self._emit(AGUIEventType.STEP_FINISHED, stepName=step_name)

    # ── Text messages ─────────────────────────────────────────

    def text_message_start(
        self,
        message_id: Optional[str] = None,
        *,
        role: str = "assistant",
    ) -> dict[str, Any]:
        if self._text_message_id is not None:
            raise AGUIError(
                "A text message is already open; call text_message_end() first"
            )
        self._text_message_id = message_id or str(uuid.uuid4())
        return self._emit(
            AGUIEventType.TEXT_MESSAGE_START,
            messageId=self._text_message_id,
            role=role,
        )

    def text_message_content(
        self,
        delta: str,
        message_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        target = message_id or self._text_message_id
        if target is None:
            raise AGUIError("text_message_content() called before text_message_start()")
        # The spec requires a non-empty delta; drop empties rather than
        # producing an invalid frame.
        if not delta:
            return None
        return self._emit(
            AGUIEventType.TEXT_MESSAGE_CONTENT,
            messageId=target,
            delta=delta,
        )

    def text_message_end(self, message_id: Optional[str] = None) -> dict[str, Any]:
        target = message_id or self._text_message_id
        if target is None:
            raise AGUIError("text_message_end() called with no open text message")
        if message_id is None or message_id == self._text_message_id:
            self._text_message_id = None
        return self._emit(AGUIEventType.TEXT_MESSAGE_END, messageId=target)

    # ── Tool calls ────────────────────────────────────────────

    def tool_call_start(
        self,
        tool_call_name: str,
        tool_call_id: Optional[str] = None,
        *,
        parent_message_id: Optional[str] = None,
    ) -> dict[str, Any]:
        call_id = tool_call_id or str(uuid.uuid4())
        self._open_tool_calls.add(call_id)
        return self._emit(
            AGUIEventType.TOOL_CALL_START,
            toolCallId=call_id,
            toolCallName=tool_call_name,
            parentMessageId=parent_message_id,
        )

    def tool_call_args(
        self,
        delta: str,
        tool_call_id: Optional[str],
    ) -> Optional[dict[str, Any]]:
        if tool_call_id not in self._open_tool_calls:
            raise AGUIError(f"Unknown tool call {tool_call_id!r}; call tool_call_start() first")
        if not delta:
            return None
        return self._emit(
            AGUIEventType.TOOL_CALL_ARGS,
            toolCallId=tool_call_id,
            delta=delta,
        )

    def tool_call_end(self, tool_call_id: Optional[str]) -> dict[str, Any]:
        if tool_call_id not in self._open_tool_calls:
            raise AGUIError(f"Unknown tool call {tool_call_id!r}")
        self._open_tool_calls.discard(tool_call_id)
        return self._emit(AGUIEventType.TOOL_CALL_END, toolCallId=tool_call_id)

    def tool_call_result(
        self,
        content: str,
        tool_call_id: str,
        *,
        message_id: Optional[str] = None,
        role: str = "tool",
    ) -> dict[str, Any]:
        return self._emit(
            AGUIEventType.TOOL_CALL_RESULT,
            messageId=message_id or str(uuid.uuid4()),
            toolCallId=tool_call_id,
            content=content,
            role=role,
        )

    # ── State ─────────────────────────────────────────────────

    def state_snapshot(self, snapshot: Any) -> dict[str, Any]:
        return self._emit(AGUIEventType.STATE_SNAPSHOT, snapshot=snapshot)

    def state_delta(self, delta: list[dict[str, Any]]) -> dict[str, Any]:
        """RFC 6902 JSON Patch operations applied to the current state."""
        return self._emit(AGUIEventType.STATE_DELTA, delta=delta)

    def messages_snapshot(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return self._emit(AGUIEventType.MESSAGES_SNAPSHOT, messages=messages)

    # ── Reasoning (chain-of-thought surface) ───────────────────

    def reasoning_start(self, message_id: str) -> dict[str, Any]:
        return self._emit(AGUIEventType.REASONING_START, messageId=message_id)

    def reasoning_message_start(
        self, message_id: Optional[str] = None
    ) -> dict[str, Any]:
        return self._emit(
            AGUIEventType.REASONING_MESSAGE_START,
            messageId=message_id or str(uuid.uuid4()),
            role="reasoning",
        )

    def reasoning_message_content(
        self, delta: str, message_id: str
    ) -> Optional[dict[str, Any]]:
        if not delta:
            return None
        return self._emit(
            AGUIEventType.REASONING_MESSAGE_CONTENT,
            messageId=message_id,
            delta=delta,
        )

    def reasoning_message_end(self, message_id: str) -> dict[str, Any]:
        return self._emit(AGUIEventType.REASONING_MESSAGE_END, messageId=message_id)

    def reasoning_end(self, message_id: str) -> dict[str, Any]:
        return self._emit(AGUIEventType.REASONING_END, messageId=message_id)

    # ── Special ───────────────────────────────────────────────

    def custom(self, name: str, value: Any) -> dict[str, Any]:
        return self._emit(AGUIEventType.CUSTOM, name=name, value=value)

    def raw(self, event: Any) -> dict[str, Any]:
        return self._emit(AGUIEventType.RAW, event=event)

    def emit(self, event_type: AGUIEventType | str, **payload: Any) -> dict[str, Any]:
        """Escape hatch for events without a dedicated helper."""
        value = event_type.value if isinstance(event_type, AGUIEventType) else str(event_type)
        if value not in _EVENT_VALUES:
            raise AGUIError(f"Unknown AG-UI event type {value!r}")
        event: dict[str, Any] = {"type": value, "timestamp": _now_ms()}
        event.update({key: val for key, val in payload.items() if val is not None})
        return event


# ─── AgentLink → AG-UI bridge ──────────────────────────────────────────────────

def _content_text(content: Any) -> str:
    """Flatten an AG-UI message content (string or multimodal parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    if content is None:
        return ""
    return str(content)


def last_user_text(messages: Iterable[Mapping[str, Any]]) -> str:
    """Return the text of the last user message in an AG-UI message list."""
    for message in reversed(list(messages)):
        if message.get("role") == "user":
            return _content_text(message.get("content"))
    return ""


def _normalize_output(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        for key in ("output", "response", "content"):
            if key in raw:
                return _content_text(raw[key])
        return json.dumps(raw, ensure_ascii=False, default=str)
    return str(raw)


def _chunk_to_event(
    emitter: AGUIEmitter, chunk: Any, message_id: str
) -> Optional[dict[str, Any]]:
    # A handler may stream pre-built AG-UI events directly.
    if isinstance(chunk, Mapping) and chunk.get("type") in _EVENT_VALUES:
        return dict(chunk)
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", "replace")
    return emitter.text_message_content(str(chunk), message_id)


def expose_node_as_ag_ui(node: Any) -> Callable[[dict[str, Any]], AsyncIterator[dict[str, Any]]]:
    """
    Wrap an ``AgentNode`` as an AG-UI handler.

    The last user message is delivered as an AgentLink ``REQUEST``; the node's
    reply (string, dict, or iterable of chunks) is streamed back as
    ``TEXT_MESSAGE_*`` events. If a streamed chunk is already an AG-UI event
    dict, it is passed through untouched, so advanced agents keep full control.

    Usage::

        app = create_ag_ui_app(expose_node_as_ag_ui(node))
    """

    async def handler(run_input: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        thread_id = str(run_input.get("threadId") or run_input.get("thread_id") or "")
        run_id = str(run_input.get("runId") or run_input.get("run_id") or uuid.uuid4())
        emitter = AGUIEmitter(thread_id, run_id, parent_run_id=run_input.get("parentRunId"))

        yield emitter.run_started(input=run_input)

        text = last_user_text(run_input.get("messages") or [])
        request = AgentMessage(
            type=MessageType.REQUEST,
            sender=AgentAddress("ag-ui-client", "ui"),
            recipient=getattr(node, "address", AgentAddress("agent", "default")),
            content=text,
        )

        try:
            raw = (
                node.invoke(request)
                if hasattr(node, "invoke")
                else node._handler(request)  # noqa: SLF001 - adapter fallback
            )
            if inspect.isawaitable(raw):
                raw = await raw

            if is_streamable(raw):
                start = emitter.text_message_start()
                message_id = start["messageId"]
                yield start
                if hasattr(raw, "__aiter__"):
                    async for chunk in raw:
                        event = _chunk_to_event(emitter, chunk, message_id)
                        if event is not None:
                            yield event
                else:
                    for chunk in raw:
                        event = _chunk_to_event(emitter, chunk, message_id)
                        if event is not None:
                            yield event
                yield emitter.text_message_end(message_id)
                yield emitter.run_finished()
            else:
                output = _normalize_output(raw)
                start = emitter.text_message_start()
                message_id = start["messageId"]
                yield start
                content = emitter.text_message_content(output, message_id)
                if content is not None:
                    yield content
                yield emitter.text_message_end(message_id)
                yield emitter.run_finished(result=output or None)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as RUN_ERROR
            if emitter.current_message_id is not None:
                yield emitter.text_message_end()
            yield emitter.run_error(str(exc), code="AGENT_ERROR")

    return handler


# ─── SSE transport ────────────────────────────────────────────────────────────

async def _iter_handler_events(
    handler: Callable[[dict[str, Any]], Any],
    run_input: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    result = handler(run_input)
    if inspect.isawaitable(result):
        result = await result

    if isinstance(result, Mapping):
        yield dict(result)
        return
    if hasattr(result, "__aiter__"):
        async for event in result:
            if event is not None:
                yield event
        return
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, Mapping)):
        for event in result:
            if event is not None:
                yield event
        return
    if result is None:
        return
    raise TypeError(
        "AG-UI handler must return an event dict or an iterable of event dicts"
    )


def _sse_encode(event: dict[str, Any]) -> bytes:
    payload = json.dumps(event, ensure_ascii=False, default=str)
    return f"data: {payload}\n\n".encode("utf-8")


def create_ag_ui_app(
    handler: Callable[[dict[str, Any]], Any],
    path: str = "/",
    *,
    cors_origin: str = "*",
):
    """
    Create an aiohttp app serving the AG-UI SSE transport.

    ``POST <path>`` accepts a ``RunAgentInput`` JSON object and replies with a
    ``text/event-stream`` body: one ``data:`` frame per AG-UI event. ``GET``
    returns endpoint metadata; ``OPTIONS`` answers CORS preflight.

    Args:
        handler: Called with the parsed ``RunAgentInput``. May return an event
            dict, an iterable of events, an async iterable, or an awaitable
            resolving to one of those. ``expose_node_as_ag_ui`` builds one for
            an ``AgentNode``.
        path: Endpoint path (default ``/``).
        cors_origin: ``Access-Control-Allow-Origin`` value (default ``*``).

    Requires ``pip install cdzzy-agentlink[agui]`` (aiohttp).
    """
    from aiohttp import web

    def _cors_headers() -> dict[str, str]:
        return {
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Accept",
        }

    async def handle_post(request: web.Request) -> web.StreamResponse:
        try:
            run_input = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"code": "INVALID_JSON", "message": "Body is not valid JSON"}},
                status=400,
                headers=_cors_headers(),
            )

        if not isinstance(run_input, dict):
            return web.json_response(
                {"error": {"code": "INVALID_RUN_INPUT", "message": "Expected a JSON object"}},
                status=400,
                headers=_cors_headers(),
            )

        if not (run_input.get("threadId") or run_input.get("thread_id")):
            return web.json_response(
                {
                    "error": {
                        "code": "INVALID_RUN_INPUT",
                        "message": "RunAgentInput requires 'threadId'",
                    }
                },
                status=400,
                headers=_cors_headers(),
            )

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                **_cors_headers(),
            },
        )
        await response.prepare(request)

        try:
            async for event in _iter_handler_events(handler, run_input):
                await response.write(_sse_encode(event))
            await response.write_eof()
        except Exception as exc:  # noqa: BLE001 - the stream is already open
            error_event = {
                "type": AGUIEventType.RUN_ERROR.value,
                "timestamp": _now_ms(),
                "message": str(exc),
                "code": "AGUI_TRANSPORT_ERROR",
            }
            await response.write(_sse_encode(error_event))
            await response.write_eof()
        return response

    async def handle_get(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "protocol": "ag-ui",
                "protocolVersion": AGUI_PROTOCOL_VERSION,
                "transport": "sse",
                "endpoint": path,
                "events": sorted(_EVENT_VALUES),
            },
            headers=_cors_headers(),
        )

    async def handle_options(request: web.Request) -> web.Response:
        return web.Response(status=204, headers=_cors_headers())

    app = web.Application()
    app.router.add_post(path, handle_post)
    app.router.add_get(path, handle_get)
    app.router.add_options(path, handle_options)
    return app


# ─── Resume payload validation ────────────────────────────────────────────────

_PYTHON_TYPES: dict[str, Any] = {
    "object": Mapping,
    "array": (list, tuple),
    "string": str,
    "boolean": bool,
    "number": (int, float),
    "integer": int,
    "null": type(None),
}


def _parse_datetime(value: str) -> Optional[datetime]:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _validate_payload(schema: Any, payload: Any) -> None:
    """Minimal JSON Schema check: object shape, required keys, primitive types."""
    if not isinstance(schema, Mapping):
        return

    expected = schema.get("type")
    if expected == "object":
        if not isinstance(payload, Mapping):
            raise AGUIResumeError("Resume payload must be an object")
        required = schema.get("required") or []
        missing = [key for key in required if key not in payload]
        if missing:
            raise AGUIResumeError(
                "Resume payload is missing required field(s): " + ", ".join(missing)
            )
        properties = schema.get("properties") or {}
        for key, subschema in properties.items():
            if key in payload and isinstance(subschema, Mapping):
                _validate_value(key, subschema, payload[key])


def _validate_value(key: str, schema: Mapping[str, Any], value: Any) -> None:
    expected = schema.get("type")
    if expected is None:
        return
    python_type = _PYTHON_TYPES.get(expected)
    if python_type is None:
        return
    if expected in ("number", "integer") and isinstance(value, bool):
        raise AGUIResumeError(f"Resume field {key!r} must be a {expected}, not a boolean")
    if expected == "object" and isinstance(value, Mapping):
        _validate_payload(schema, value)
        return
    if not isinstance(value, python_type):
        raise AGUIResumeError(f"Resume field {key!r} must be a {expected}")
