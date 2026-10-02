"""
Tests for the A2A v1.0 SSE streaming layer:
- stream event construction (Task / status-update / artifact-update)
- stream event -> AgentMessage decoding (envelope-aware)
- A2ATransport streaming client (build_stream_request / parse_stream_events)
- A2ATransport streaming server (parse_stream_request / build_stream_response)
- full server -> client round-trip over the SSE wire format
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import pytest

from agentlink.a2a import (
    ENVELOPE_METADATA_KEY,
    EVENT_KIND_ARTIFACT_UPDATE,
    EVENT_KIND_STATUS_UPDATE,
    EVENT_KIND_TASK,
    TASK_STATE_COMPLETED,
    TASK_STATE_SUBMITTED,
    TASK_STATE_WORKING,
    A2ATransport,
    A2ATransportError,
    artifact_update_event,
    status_update_event,
    stream_event_to_agent_message,
    task_object,
)
from agentlink.protocol.message import AgentAddress, AgentMessage, MessageType


def make_message(**overrides) -> AgentMessage:
    defaults = dict(
        type=MessageType.REQUEST,
        sender=AgentAddress("planner", "my-app", capability="web-search"),
        recipient=AgentAddress("researcher", "my-app"),
        content="Analyze Q1 market data",
        correlation_id="corr-123",
        metadata={"user_tag": "u1"},
    )
    defaults.update(overrides)
    return AgentMessage(**defaults)


def make_chunk(text: str, seq: int = 0, **overrides) -> AgentMessage:
    """Build a streamed reply chunk (what a server would emit per token)."""
    defaults = dict(
        type=MessageType.REPLY,
        sender=AgentAddress("researcher", "my-app"),
        recipient=AgentAddress("planner", "my-app"),
        content=text,
        correlation_id="corr-123",
        metadata={"seq": seq},
    )
    defaults.update(overrides)
    return AgentMessage(**defaults)


SERVER = A2ATransport(local_agent_id="researcher")


# ── Stream event construction (mapping layer) ────────────────────────────────

class TestStreamEventConstruction:

    def test_event_kind_constants(self):
        assert EVENT_KIND_TASK == "task"
        assert EVENT_KIND_STATUS_UPDATE == "status-update"
        assert EVENT_KIND_ARTIFACT_UPDATE == "artifact-update"

    def test_task_object_opens_the_stream(self):
        msg = make_message()
        task = task_object(msg)
        assert task["kind"] == "task"
        assert task["status"]["state"] == TASK_STATE_SUBMITTED
        assert task["id"] and task["contextId"]

    def test_status_update_event_pure_state(self):
        event = status_update_event(
            None, TASK_STATE_WORKING, task_id="t-1", context_id="c-1"
        )
        assert event["kind"] == "status-update"
        assert event["taskId"] == "t-1"
        assert event["contextId"] == "c-1"
        assert event["status"]["state"] == TASK_STATE_WORKING
        assert "message" not in event["status"]
        assert event["final"] is False

    def test_status_update_event_final_flag(self):
        event = status_update_event(
            None, TASK_STATE_COMPLETED, task_id="t-1", context_id="c-1", final=True
        )
        assert event["final"] is True
        assert event["status"]["state"] == TASK_STATE_COMPLETED

    def test_status_update_event_carries_message_envelope(self):
        msg = make_chunk("partial answer", seq=1)
        event = status_update_event(
            msg, TASK_STATE_WORKING, task_id="t-1", context_id="c-1"
        )
        embedded = event["status"]["message"]
        assert embedded["kind"] == "message"
        assert ENVELOPE_METADATA_KEY in embedded["metadata"]  # lossless envelope

    def test_artifact_update_event_fields(self):
        event = artifact_update_event(
            "chunk text", task_id="t-1", context_id="c-1",
            metadata={"seq": 2}, last_chunk=True,
        )
        assert event["kind"] == "artifact-update"
        assert event["taskId"] == "t-1"
        assert event["contextId"] == "c-1"
        assert event["append"] is False
        assert event["lastChunk"] is True
        parts = event["artifact"]["parts"]
        assert parts and parts[0]["kind"] == "text"
        assert event["artifact"]["metadata"] == {"seq": 2}

    def test_artifact_update_event_append_semantics(self):
        event = artifact_update_event(
            "more", task_id="t-1", context_id="c-1",
            append=True, last_chunk=False, artifact_id="a-1",
        )
        assert event["append"] is True
        assert event["lastChunk"] is False
        assert event["artifact"]["artifactId"] == "a-1"


# ── Stream event decoding (mapping layer) ────────────────────────────────────

class TestStreamEventDecoding:

    def test_pure_status_update_decodes_to_none(self):
        event = status_update_event(None, TASK_STATE_WORKING)
        assert stream_event_to_agent_message(event) is None

    def test_artifact_update_decodes_to_reply(self):
        event = artifact_update_event(
            "chunk text", task_id="t-1", context_id="c-1", metadata={"seq": 3}
        )
        msg = stream_event_to_agent_message(event)
        assert msg is not None
        assert msg.type is MessageType.REPLY
        assert msg.content == "chunk text"
        assert msg.metadata.get("seq") == 3

    def test_status_update_with_message_is_envelope_lossless(self):
        chunk = make_chunk("streamed payload", seq=4,
                           metadata={"seq": 4, "user_tag": "u1"})
        event = status_update_event(chunk, TASK_STATE_WORKING)
        decoded = stream_event_to_agent_message(event)
        assert decoded is not None
        assert decoded.content == "streamed payload"
        assert decoded.metadata.get("user_tag") == "u1"
        assert decoded.metadata.get("seq") == 4
        assert decoded.sender.agent_id == "researcher"

    def test_unknown_kind_returns_none(self):
        assert stream_event_to_agent_message({"kind": "mystery"}) is None


# ── Streaming transport: client side ─────────────────────────────────────────

class TestStreamClient:

    def test_build_stream_request_uses_v1_method(self):
        msg = make_message()
        request = A2ATransport(local_agent_id="planner").build_stream_request(msg)
        assert request["method"] == "SendStreamingMessage"
        assert request["params"]["message"]["kind"] == "message"

    def test_parse_stream_events_roundtrip(self):
        chunks = [make_chunk("alpha", 0), make_chunk("beta", 1), make_chunk("gamma", 2)]
        body = SERVER.build_stream_response(chunks, request_id="req-1")
        replies = SERVER.parse_stream_events(body, sent=make_message())
        assert [r.content for r in replies] == ["alpha", "beta", "gamma"]
        assert all(r.type is MessageType.REPLY for r in replies)
        assert all(r.metadata.get("seq") == i for i, r in enumerate(replies))

    def test_parse_stream_events_accepts_bytes(self):
        body = SERVER.build_stream_response([make_chunk("bytes!", 0)], request_id=1)
        replies = SERVER.parse_stream_events(body.encode("utf-8"))
        assert replies and replies[0].content == "bytes!"

    def test_parse_stream_events_skips_keepalives(self):
        chunk = make_chunk("with keepalive", 0)
        body = SERVER.build_stream_response([chunk], request_id=1)
        body = ": ping\n\n" + body + ": ping\n\n"
        replies = SERVER.parse_stream_events(body)
        assert [r.content for r in replies] == ["with keepalive"]

    def test_parse_stream_events_fills_correlation_from_sent(self):
        # A plain streamed Message (no contextId) gets its correlation id
        # filled in from the outgoing message, mirroring parse_response.
        sent = make_message()
        event = artifact_update_event("hi", task_id="t-1", context_id="ctx-9")
        body = (
            'data: {"jsonrpc":"2.0","id":1,"result":'
            + json.dumps(event) + "}\n\n"
        )
        replies = SERVER.parse_stream_events(body, sent=sent)
        assert replies[0].correlation_id == "ctx-9"  # wire value wins

    def test_message_event_gets_sent_correlation(self):
        sent = make_message()
        payload = {"kind": "message", "messageId": "m-1", "role": "ROLE_AGENT",
                   "parts": [{"kind": "text", "text": "direct"}]}
        body = 'data: {"jsonrpc":"2.0","id":1,"result":' + json.dumps(payload) + '}\n\n'
        replies = SERVER.parse_stream_events(body, sent=sent)
        assert replies[0].type is MessageType.REPLY
        assert replies[0].correlation_id == sent.id

    def test_plain_json_error_body_raises(self):
        body = json.dumps({"jsonrpc": "2.0", "id": 1,
                           "error": {"code": -32601, "message": "Method not found"}})
        with pytest.raises(A2ATransportError, match="-32601"):
            SERVER.parse_stream_events(body)

    def test_garbage_body_raises(self):
        with pytest.raises(A2ATransportError):
            SERVER.parse_stream_events("<html>Bad Gateway</html>")

    def test_sse_error_event_raises(self):
        body = 'data: {"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"boom"}}\n\n'
        with pytest.raises(A2ATransportError, match="boom"):
            SERVER.parse_stream_events(body)

    def test_multiline_data_block_is_joined(self):
        event = artifact_update_event("joined", task_id="t", context_id="c")
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "result": event})
        # split on a safe boundary (between JSON fields, never inside an
        # escape sequence) and rejoin via SSE multi-line data semantics
        split_at = payload.index('", "') + 3
        body = f"data: {payload[:split_at]}\ndata: {payload[split_at:]}\n\n"
        replies = SERVER.parse_stream_events(body)
        assert [r.content for r in replies] == ["joined"]


# ── Streaming transport: server side ─────────────────────────────────────────

class TestStreamServer:

    def test_parse_stream_request_accepts_v1_and_legacy(self):
        msg = make_message()
        for method in ("SendStreamingMessage", "message/stream"):
            request = SERVER.build_request(msg, request_id="r1")
            request["method"] = method
            decoded = SERVER.parse_stream_request(request)
            assert decoded.content == msg.content

    def test_parse_request_rejects_stream_method(self):
        msg = make_message()
        request = SERVER.build_stream_request(msg)
        with pytest.raises(A2ATransportError):
            SERVER.parse_request(request)

    def test_parse_stream_request_rejects_send_method(self):
        msg = make_message()
        request = SERVER.build_request(msg)
        with pytest.raises(A2ATransportError):
            SERVER.parse_stream_request(request)

    def test_single_chunk_event_pattern(self):
        events = SERVER.build_stream_events([make_chunk("only", 0)], request_id="r")
        kinds = [e["result"]["kind"] for e in events]
        # Task -> WORKING -> artifact-update -> final status-update
        assert kinds == ["task", "status-update", "artifact-update", "status-update"]
        assert events[1]["result"]["status"]["state"] == TASK_STATE_WORKING
        assert events[-1]["result"]["final"] is True
        assert events[2]["result"]["lastChunk"] is True

    def test_multi_chunk_event_pattern(self):
        chunks = [make_chunk(t, i) for i, t in enumerate(["a", "b", "c"])]
        events = SERVER.build_stream_events(chunks, request_id="r")
        kinds = [e["result"]["kind"] for e in events]
        # Task -> W -> artifact(a) -> W -> artifact(b) -> W -> artifact(c last) -> final
        assert kinds == ["task", "status-update", "artifact-update",
                         "status-update", "artifact-update",
                         "status-update", "artifact-update", "status-update"]
        states = [e["result"]["status"]["state"]
                  for e in events if e["result"]["kind"] == "status-update"]
        assert states == [TASK_STATE_WORKING, TASK_STATE_WORKING,
                          TASK_STATE_WORKING, TASK_STATE_COMPLETED]
        last_chunks = [e["result"]["lastChunk"]
                       for e in events if e["result"]["kind"] == "artifact-update"]
        assert last_chunks == [False, False, True]
        # pure WORKING events carry no message payload (no duplicated content)
        for e in events:
            if (e["result"]["kind"] == "status-update"
                    and not e["result"]["final"]):
                assert "message" not in e["result"]["status"]

    def test_empty_chunks_still_close_the_stream(self):
        events = SERVER.build_stream_events([], request_id="r")
        kinds = [e["result"]["kind"] for e in events]
        assert kinds == ["task", "status-update"]
        assert events[-1]["result"]["final"] is True

    def test_sse_body_format(self):
        events = SERVER.build_stream_events([make_chunk("x", 0)], request_id="r")
        body = SERVER.build_sse_body(events)
        assert body.endswith("\n\n")
        data_blocks = [line for line in body.splitlines() if line.startswith("data: ")]
        assert len(data_blocks) == len(events)
        for line in data_blocks:
            json.loads(line[6:])  # each data block is valid JSON

    def test_build_stream_response_content_type_hint(self):
        body = SERVER.build_stream_response([make_chunk("y", 0)], request_id="r")
        assert "data: " in body


# ── Full wire round-trip (server build -> client parse) ──────────────────────

class TestStreamRoundTrip:

    def test_multi_chunk_wire_roundtrip_is_ordered(self):
        texts = ["The", " quick", " brown", " fox"]
        chunks = [make_chunk(t, i) for i, t in enumerate(texts)]
        body = SERVER.build_stream_response(chunks, request_id="wire-1")
        client = A2ATransport(local_agent_id="planner")
        replies = client.parse_stream_events(body, sent=make_message())
        assert "".join(r.content for r in replies) == "The quick brown fox"
        assert [r.metadata.get("seq") for r in replies] == [0, 1, 2, 3]

    def test_roundtrip_preserves_envelope_metadata(self):
        chunk = make_chunk("rich", 0, metadata={"user_tag": "u1",
                                                "nested": {"k": [1, 2, {"deep": True}]}})
        body = SERVER.build_stream_response([chunk], request_id="wire-2")
        replies = SERVER.parse_stream_events(body)
        assert replies[0].metadata.get("nested") == {"k": [1, 2, {"deep": True}]}

    def test_request_response_ids_align(self):
        chunks = [make_chunk("z", 0)]
        body = SERVER.build_stream_response(chunks, request_id=42)
        for block in body.split("\n\n"):
            if not block.strip():
                continue
            payload = json.loads(block[len("data: "):])
            assert payload["id"] == 42
