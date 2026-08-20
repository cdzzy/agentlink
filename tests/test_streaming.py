"""
Tests for streaming support (v0.3.0).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agentlink.runtime.node import AgentNode
from agentlink.runtime.bus import AgentBus
from agentlink.runtime.stream import StreamResult, is_streamable, stream_message
from agentlink.protocol.message import MessageType, AgentAddress


def make_bus_with(handler):
    bus = AgentBus()
    node = AgentNode("worker", handler=handler)
    bus.register(node)
    return bus, node


class TestStreamHelpers:

    def test_is_streamable(self):
        assert is_streamable(["a", "b"]) is True
        assert is_streamable(iter(["a"])) is True
        assert is_streamable("text") is False
        assert is_streamable({"a": 1}) is False
        assert is_streamable(None) is False

    def test_stream_result_collect(self):
        r = StreamResult("corr-1")
        r.push("hello ")
        r.push("world")
        r.finish()
        assert r.collect() == "hello world"
        assert list(r.chunks()) == ["hello ", "world"]

    def test_stream_result_iteration(self):
        r = StreamResult("corr-2")
        r.push("a")
        r.push("b")
        r.finish()
        assert list(r) == ["a", "b"]

    def test_stream_result_failure(self):
        r = StreamResult("corr-3")
        r.fail("boom")
        with pytest.raises(RuntimeError, match="boom"):
            r.collect()

    def test_stream_message_builder(self):
        m = stream_message(MessageType.STREAM_CHUNK, "c1", AgentAddress("a"), AgentAddress("b"), "x")
        assert m.type == MessageType.STREAM_CHUNK
        assert m.correlation_id == "c1"
        assert m.content == "x"


class TestStreaming:

    def test_streaming_response(self):
        def handler(message):
            for part in ["Hello", " there, ", "world"]:
                yield part

        bus, worker = make_bus_with(handler)
        client = AgentNode("client", handler=lambda m: "")
        bus.register(client)

        stream = client.stream("worker", "greet me")
        parts = list(stream)
        assert parts == ["Hello", " there, ", "world"]
        assert stream.collect() == "Hello there, world"

    def test_streaming_with_list_handler(self):
        def handler(message):
            return ["chunk1", "chunk2", "chunk3"]

        bus, worker = make_bus_with(handler)
        client = AgentNode("client", handler=lambda m: "")
        bus.register(client)

        stream = client.stream("worker", "data please")
        assert stream.collect() == "chunk1chunk2chunk3"

    def test_normal_request_still_works(self):
        def handler(message):
            return "plain reply"

        bus, worker = make_bus_with(handler)
        client = AgentNode("client", handler=lambda m: "")
        bus.register(client)

        reply = client.send("worker", "hi")
        assert reply.type == MessageType.REPLY
        assert reply.content == "plain reply"

    def test_worker_can_stream_back_multiple_requests(self):
        def handler(message):
            return [message.content, "!!"]

        bus, worker = make_bus_with(handler)
        client = AgentNode("client", handler=lambda m: "")
        bus.register(client)

        s1 = client.stream("worker", "first")
        s2 = client.stream("worker", "second")
        assert s1.collect() == "first!!"
        assert s2.collect() == "second!!"

    def test_handler_error_returns_error_message(self):
        def handler(message):
            raise RuntimeError("handler crashed")

        bus, worker = make_bus_with(handler)
        client = AgentNode("client", handler=lambda m: "")
        bus.register(client)

        reply = client.send("worker", "hi")
        assert reply.type == MessageType.ERROR
        assert "handler crashed" in str(reply.content)