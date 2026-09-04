"""
Tests for the engram memory integration (v0.5.0).

Runs against a fake engram-mcp stdio server (pure Python, no Node needed)
that implements the same JSON-RPC wire protocol as the real `engram-mcp`.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from agentlink.integrations.engram import (
    EngramMCPClient,
    EngramMCPError,
    EngramMemoryBackend,
    attach_memory,
)
from agentlink.runtime.node import AgentNode
from agentlink.runtime.bus import AgentBus

FAKE_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_engram_mcp.py")


@pytest.fixture
def client():
    client = EngramMCPClient(command=sys.executable, args=[FAKE_SERVER]).start()
    yield client
    client.stop()


class TestEngramMCPClient:

    def test_store_and_get(self, client):
        stored = client.store("User prefers dark mode", importance="high", tags=["pref"])
        assert stored["id"] == "mem-1"
        assert stored["content"] == "User prefers dark mode"

        fetched = client.get("mem-1")
        assert fetched is not None
        assert fetched["importance"] == "high"

    def test_recall_finds_by_keywords(self, client):
        client.store("Solar costs fell 12% YoY")
        client.store("Wind permits accelerated")
        hits = client.recall("solar costs", limit=5)
        assert len(hits) >= 1
        assert any("Solar" in h["content"] for h in hits)

    def test_forget_removes(self, client):
        stored = client.store("temporary fact")
        client.forget(stored["id"])
        assert client.get(stored["id"]) is None

    def test_stats(self, client):
        client.store("one")
        client.store("two")
        stats = client.stats()
        assert stats["total"] == 2

    def test_not_started_raises(self):
        client = EngramMCPClient(command=sys.executable, args=[FAKE_SERVER])
        with pytest.raises(EngramMCPError, match="not started"):
            client.store("x")


class TestMemoryBackend:

    def test_record_message_stores_episodic_memory(self, client):
        from agentlink.protocol.message import AgentMessage, MessageType, AgentAddress

        backend = EngramMemoryBackend(client, namespace="fleet")
        msg = AgentMessage(
            type=MessageType.REQUEST,
            sender=AgentAddress("planner"),
            recipient=AgentAddress("worker"),
            content="Research AI trends",
        )
        stored = backend.record_message(msg)
        assert stored is not None
        assert "planner" in stored["content"]
        assert stored["namespace"] == "fleet"

    def test_recall_context(self, client):
        backend = EngramMemoryBackend(client)
        client.store("The budget meeting is on Friday", importance="high")
        hits = backend.recall_context("budget meeting")
        assert len(hits) >= 1

    def test_record_failure_never_raises(self, client):
        backend = EngramMemoryBackend(client)
        client.stop()  # server gone
        from agentlink.protocol.message import AgentMessage, MessageType, AgentAddress
        msg = AgentMessage(type=MessageType.REQUEST, sender=AgentAddress("a"),
                           recipient=AgentAddress("b"), content="x")
        assert backend.record_message(msg) is None  # swallowed


class TestBusIntegration:

    def test_attach_memory_records_routed_messages(self, client):
        bus = AgentBus()
        bus.register(AgentNode("worker", handler=lambda m: "ok"))
        backend = attach_memory(bus, client=client)

        bus.send("client", "worker", "hello fleet")

        hits = backend.recall_context("hello fleet")
        assert len(hits) >= 1

    def test_middleware_passes_message_through(self, client):
        bus = AgentBus()
        bus.register(AgentNode("worker", handler=lambda m: "fine"))
        backend = attach_memory(bus, client=client)

        reply = bus.send("client", "worker", "check")
        assert reply is not None
        assert reply.content == "fine"
