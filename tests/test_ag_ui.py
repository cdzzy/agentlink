"""
Tests for the AG-UI adapter (events, interrupts, SSE transport, node bridge).

The emitter, interrupt manager and node bridge are dependency-free; the
transport tests need aiohttp and are skipped when it is not installed.
"""

import json

import pytest

from agentlink.adapters.ag_ui import (
    AGUI_APPROVAL_SCHEMA,
    AGUI_PROTOCOL_VERSION,
    AGUIEmitter,
    AGUIError,
    AGUIEventType,
    AGUIInterruptManager,
    AGUIResume,
    AGUIResumeError,
    create_ag_ui_app,
    expose_node_as_ag_ui,
    last_user_text,
)
from agentlink.runtime.node import AgentNode

try:
    from aiohttp import test_utils

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


# ─── Event emitter ─────────────────────────────────────────────────────────────

class TestEmitterLifecycle:
    def test_run_started(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        event = emitter.run_started()
        assert event["type"] == "RUN_STARTED"
        assert event["threadId"] == "thread-1"
        assert event["runId"] == "run-1"
        assert "parentRunId" not in event
        assert isinstance(event["timestamp"], int)

    def test_run_started_with_parent_and_input(self):
        emitter = AGUIEmitter("thread-1", "run-2", parent_run_id="run-1")
        event = emitter.run_started(input={"messages": []})
        assert event["parentRunId"] == "run-1"
        assert event["input"] == {"messages": []}

    def test_run_finished_and_error(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        finished = emitter.run_finished(result="ok", usage=[{"inputTokens": 3}])
        assert finished["type"] == "RUN_FINISHED"
        assert finished["result"] == "ok"
        assert finished["usage"][0]["inputTokens"] == 3

        error = emitter.run_error("boom", code="E1")
        assert error["type"] == "RUN_ERROR"
        assert error["message"] == "boom"
        assert error["code"] == "E1"

    def test_run_finished_interrupt_outcome(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        outcome = {"type": "interrupt", "interrupts": [{"id": "i-1", "reason": "confirmation"}]}
        event = emitter.run_finished(outcome=outcome)
        assert event["outcome"] == outcome

    def test_step_events(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        assert emitter.step_started("plan")["stepName"] == "plan"
        assert emitter.step_finished("plan")["type"] == "STEP_FINISHED"


class TestEmitterText:
    def test_message_triad(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        start = emitter.text_message_start()
        message_id = start["messageId"]
        assert start["role"] == "assistant"

        content = emitter.text_message_content("Hello", message_id)
        assert content["messageId"] == message_id
        assert content["delta"] == "Hello"

        end = emitter.text_message_end()
        assert end["messageId"] == message_id
        assert emitter.current_message_id is None

    def test_empty_delta_is_dropped(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        emitter.text_message_start()
        assert emitter.text_message_content("") is None

    def test_content_requires_start(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        with pytest.raises(AGUIError):
            emitter.text_message_content("orphan")

    def test_end_requires_start(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        with pytest.raises(AGUIError):
            emitter.text_message_end()

    def test_second_message_must_close_first(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        emitter.text_message_start()
        with pytest.raises(AGUIError):
            emitter.text_message_start()


class TestEmitterTools:
    def test_tool_call_sequence(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        start = emitter.tool_call_start("search", "tc-1", parent_message_id="m-1")
        assert start["toolCallId"] == "tc-1"
        assert start["toolCallName"] == "search"
        assert start["parentMessageId"] == "m-1"

        args = emitter.tool_call_args('{"q":', "tc-1")
        assert args["delta"] == '{"q":'

        end = emitter.tool_call_end("tc-1")
        assert end["type"] == "TOOL_CALL_END"

        result = emitter.tool_call_result("found", "tc-1", message_id="m-2")
        assert result["role"] == "tool"
        assert result["toolCallId"] == "tc-1"
        assert result["messageId"] == "m-2"

    def test_unknown_tool_call(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        with pytest.raises(AGUIError):
            emitter.tool_call_args("{}", "missing")
        with pytest.raises(AGUIError):
            emitter.tool_call_end("missing")


class TestEmitterStateAndExtras:
    def test_state_events(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        assert emitter.state_snapshot({"step": 1})["snapshot"] == {"step": 1}
        assert emitter.state_delta([{"op": "replace", "path": "/step", "value": 2}])[
            "delta"
        ][0]["op"] == "replace"
        assert emitter.messages_snapshot([{"id": "m", "role": "user"}])["messages"][0][
            "role"
        ] == "user"

    def test_reasoning_events(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        start = emitter.reasoning_message_start()
        assert start["role"] == "reasoning"
        content = emitter.reasoning_message_content("thinking", start["messageId"])
        assert content["delta"] == "thinking"
        assert emitter.reasoning_message_end(start["messageId"])["type"] == (
            "REASONING_MESSAGE_END"
        )

    def test_custom_raw_and_escape_hatch(self):
        emitter = AGUIEmitter("thread-1", "run-1")
        assert emitter.custom("progress", {"pct": 50})["name"] == "progress"
        assert emitter.raw({"native": True})["event"] == {"native": True}
        event = emitter.emit(AGUIEventType.STEP_STARTED, stepName="x")
        assert event["type"] == "STEP_STARTED"
        with pytest.raises(AGUIError):
            emitter.emit("NOT_AN_EVENT")

    def test_unknown_event_type_enum_is_complete(self):
        assert AGUIEventType.TEXT_MESSAGE_CONTENT.value == "TEXT_MESSAGE_CONTENT"
        assert AGUI_PROTOCOL_VERSION == "1.0"


# ─── Interrupts ────────────────────────────────────────────────────────────────

class TestInterrupts:
    def test_create_and_outcome(self):
        manager = AGUIInterruptManager()
        interrupt = manager.create(
            reason="tool_call",
            message="Send this email?",
            tool_call_id="tc-1",
            response_schema=AGUI_APPROVAL_SCHEMA,
        )
        outcome = manager.outcome()
        assert outcome is not None
        assert outcome["type"] == "interrupt"
        payload = outcome["interrupts"][0]
        assert payload["id"] == interrupt.id
        assert payload["reason"] == "tool_call"
        assert payload["toolCallId"] == "tc-1"
        assert payload["responseSchema"] == AGUI_APPROVAL_SCHEMA

    def test_outcome_is_none_when_no_interrupts(self):
        assert AGUIInterruptManager().outcome() is None

    def test_resolve_approval(self):
        manager = AGUIInterruptManager()
        interrupt = manager.create(response_schema=AGUI_APPROVAL_SCHEMA)
        resolved = manager.resolve(
            {"interruptId": interrupt.id, "status": "resolved", "payload": {"approved": True}}
        )
        assert resolved.id == interrupt.id
        assert len(manager) == 0

    def test_resolve_cancelled_skips_schema(self):
        manager = AGUIInterruptManager()
        interrupt = manager.create(response_schema=AGUI_APPROVAL_SCHEMA)
        manager.resolve({"interruptId": interrupt.id, "status": "cancelled"})
        assert len(manager) == 0

    def test_unknown_interrupt(self):
        manager = AGUIInterruptManager()
        with pytest.raises(AGUIResumeError):
            manager.resolve({"interruptId": "nope", "status": "resolved"})

    def test_missing_required_field(self):
        manager = AGUIInterruptManager()
        interrupt = manager.create(response_schema=AGUI_APPROVAL_SCHEMA)
        with pytest.raises(AGUIResumeError, match="approved"):
            manager.resolve({"interruptId": interrupt.id, "payload": {}})
        assert interrupt.id in [item.id for item in manager.open]

    def test_wrong_payload_type(self):
        manager = AGUIInterruptManager()
        interrupt = manager.create(response_schema=AGUI_APPROVAL_SCHEMA)
        with pytest.raises(AGUIResumeError, match="boolean"):
            manager.resolve(
                {"interruptId": interrupt.id, "payload": {"approved": "yes"}}
            )

    def test_expired_interrupt(self):
        manager = AGUIInterruptManager()
        interrupt = manager.create(expires_at="2020-01-01T00:00:00Z")
        with pytest.raises(AGUIResumeError, match="expired"):
            manager.resolve({"interruptId": interrupt.id, "payload": {"approved": True}})
        assert len(manager) == 0

    def test_resume_dataclass(self):
        resume = AGUIResume.from_dict(
            {"interruptId": "i-1", "status": "resolved", "payload": {"approved": True}}
        )
        assert resume.interrupt_id == "i-1"
        assert resume.to_dict()["payload"] == {"approved": True}

    def test_resume_validation(self):
        with pytest.raises(AGUIResumeError):
            AGUIResume.from_dict({"status": "resolved"})
        with pytest.raises(AGUIResumeError):
            AGUIResume(interrupt_id="i", status="unknown")
        resumes = AGUIResume.parse_many(
            [{"interruptId": "a"}, {"interruptId": "b", "status": "cancelled"}]
        )
        assert [r.interrupt_id for r in resumes] == ["a", "b"]


# ─── Node bridge (no transport) ────────────────────────────────────────────────

def _run_input(messages=None, thread_id="thread-1", run_id="run-1"):
    return {
        "threadId": thread_id,
        "runId": run_id,
        "messages": messages or [{"id": "m-1", "role": "user", "content": "Hi"}],
    }


class TestLastUserText:
    def test_plain_string(self):
        messages = [
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        assert last_user_text(messages) == "b"

    def test_multimodal_parts(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ],
            }
        ]
        assert last_user_text(messages) == "Hello world"

    def test_no_user_message(self):
        assert last_user_text([{"role": "assistant", "content": "x"}]) == ""


class TestExposeNode:
    async def test_streaming_handler(self):
        node = AgentNode("assistant", lambda message: ["Hel", "lo"])
        handler = expose_node_as_ag_ui(node)

        events = [event async for event in handler(_run_input())]
        types = [event["type"] for event in events]

        assert types[0] == "RUN_STARTED"
        assert "TEXT_MESSAGE_START" in types
        assert types.count("TEXT_MESSAGE_CONTENT") == 2
        assert types[-1] == "RUN_FINISHED"

        deltas = [e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT"]
        assert "".join(deltas) == "Hello"

    async def test_plain_string_result(self):
        node = AgentNode("assistant", lambda message: "done")
        handler = expose_node_as_ag_ui(node)

        events = [event async for event in handler(_run_input())]
        finished = events[-1]
        assert finished["type"] == "RUN_FINISHED"
        assert finished["result"] == "done"

    async def test_dict_result(self):
        node = AgentNode("assistant", lambda message: {"output": "ok", "extra": 1})
        handler = expose_node_as_ag_ui(node)
        events = [event async for event in handler(_run_input())]
        assert events[-1]["result"] == "ok"

    async def test_prebuilt_events_pass_through(self):
        def stream(message):
            yield {"type": "STEP_STARTED", "stepName": "custom"}

        node = AgentNode("assistant", stream)
        handler = expose_node_as_ag_ui(node)
        events = [event async for event in handler(_run_input())]
        assert any(
            e["type"] == "STEP_STARTED" and e["stepName"] == "custom" for e in events
        )

    async def test_handler_error_becomes_run_error(self):
        def boom(message):
            raise ValueError("kaboom")

        node = AgentNode("assistant", boom)
        handler = expose_node_as_ag_ui(node)
        events = [event async for event in handler(_run_input())]
        types = [event["type"] for event in events]

        assert types[0] == "RUN_STARTED"
        assert types[-1] == "RUN_ERROR"
        assert events[-1]["code"] == "AGENT_ERROR"
        assert "kaboom" in events[-1]["message"]

    async def test_async_handler(self):
        async def reply(message):
            return "async reply"

        node = AgentNode("assistant", reply)
        handler = expose_node_as_ag_ui(node)
        events = [event async for event in handler(_run_input())]
        assert events[-1]["result"] == "async reply"

    async def test_node_invoke_bypasses_bus(self):
        node = AgentNode("assistant", lambda message: message.content.upper())
        from agentlink.protocol.message import AgentAddress, AgentMessage, MessageType

        message = AgentMessage(
            type=MessageType.REQUEST,
            sender=AgentAddress("client", "ui"),
            recipient=node.address,
            content="hi",
        )
        assert node.invoke(message) == "HI"


# ─── SSE transport ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
class TestTransport:
    async def test_streams_run_lifecycle(self):
        async def handler(run_input):
            emitter = AGUIEmitter(run_input["threadId"], run_input["runId"])
            yield emitter.run_started()
            start = emitter.text_message_start()
            yield start
            content = emitter.text_message_content("Hello", start["messageId"])
            yield content
            yield emitter.text_message_end()
            yield emitter.run_finished(result="Hello")

        app = create_ag_ui_app(handler)

        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.post("/", json=_run_input())
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            body = await resp.text()

        events = _parse_sse(body)
        types = [event["type"] for event in events]
        assert types == [
            "RUN_STARTED",
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TEXT_MESSAGE_END",
            "RUN_FINISHED",
        ]
        assert events[2]["delta"] == "Hello"
        assert events[-1]["result"] == "Hello"

    async def test_handler_may_return_list(self):
        emitter = AGUIEmitter("thread-1", "run-1")

        def handler(run_input):
            return [emitter.run_started(), emitter.run_finished(result="ok")]

        app = create_ag_ui_app(handler)
        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.post("/", json=_run_input())
            body = await resp.text()

        events = _parse_sse(body)
        assert [event["type"] for event in events] == ["RUN_STARTED", "RUN_FINISHED"]

    async def test_invalid_json_is_400(self):
        async def handler(run_input):
            yield {"type": "RUN_STARTED"}

        app = create_ag_ui_app(handler)
        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.post("/", data="not json")
            assert resp.status == 400
            body = await resp.json()
            assert body["error"]["code"] == "INVALID_JSON"

    async def test_missing_thread_id_is_400(self):
        async def handler(run_input):
            yield {"type": "RUN_STARTED"}

        app = create_ag_ui_app(handler)
        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.post("/", json={"runId": "run-1", "messages": []})
            assert resp.status == 400
            body = await resp.json()
            assert body["error"]["code"] == "INVALID_RUN_INPUT"

    async def test_handler_exception_emits_run_error(self):
        async def handler(run_input):
            yield {"type": "RUN_STARTED", "threadId": "t", "runId": "r"}
            raise RuntimeError("stream broke")

        app = create_ag_ui_app(handler)
        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.post("/", json=_run_input())
            body = await resp.text()

        events = _parse_sse(body)
        assert events[-1]["type"] == "RUN_ERROR"
        assert events[-1]["code"] == "AGUI_TRANSPORT_ERROR"
        assert "stream broke" in events[-1]["message"]

    async def test_get_metadata_and_options(self):
        async def handler(run_input):
            yield {"type": "RUN_STARTED"}

        app = create_ag_ui_app(handler)
        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.get("/")
            body = await resp.json()
            assert body["protocol"] == "ag-ui"
            assert body["protocolVersion"] == "1.0"
            assert "TEXT_MESSAGE_CONTENT" in body["events"]

            preflight = await client.options("/")
            assert preflight.status == 204
            assert preflight.headers["Access-Control-Allow-Origin"] == "*"

    async def test_custom_endpoint_path(self):
        async def handler(run_input):
            yield {"type": "RUN_STARTED", "threadId": "t", "runId": "r"}

        app = create_ag_ui_app(handler, path="/agui")
        async with test_utils.TestClient(test_utils.TestServer(app)) as client:
            resp = await client.post("/agui", json=_run_input())
            assert resp.status == 200
            body = await resp.text()
            assert _parse_sse(body)[0]["type"] == "RUN_STARTED"
