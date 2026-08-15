"""
Tests for v0.2.0 features:
- Message schemas with validation (#1)
- Dead letter queue (#2)
- Message encryption (#5)
- Protocol gateway (#6)
- WebSocket transport (#4)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass

import pytest

from agentlink.protocol.message import AgentMessage, MessageType, AgentAddress
from agentlink.runtime.node import AgentNode
from agentlink.runtime.bus import AgentBus
from agentlink.schemas import MessageSchema, SchemaRegistry
from agentlink.dlq import DeadLetterQueue, DeadLetter
from agentlink.gateway import ProtocolGateway

try:
    from agentlink.security import MessageEncryptor, generate_key, encrypt_message, decrypt_message
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

try:
    import websockets  # noqa: F401
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

from agentlink.transport import serialize_message, deserialize_message


# ── Message Schemas (#1) ────────────────────────────────────────────────

@dataclass
class TaskMessage(MessageSchema):
    task_id: str
    priority: int = 1


class TestMessageSchemas:

    def test_dataclass_schema_coerce_dict(self):
        task = TaskMessage.coerce({"task_id": "t-001", "priority": 5})
        assert isinstance(task, TaskMessage)
        assert task.task_id == "t-001"
        assert task.priority == 5

    def test_dataclass_schema_passthrough(self):
        task = TaskMessage(task_id="t-002")
        assert TaskMessage.coerce(task) is task

    def test_schema_registry_register_and_validate(self):
        registry = SchemaRegistry()
        registry.register("task", TaskMessage)
        result = registry.validate("task", {"task_id": "t-9"})
        assert result.task_id == "t-9"

    def test_schema_registry_unknown_raises(self):
        registry = SchemaRegistry()
        with pytest.raises(ValueError):
            registry.validate("nope", {})

    def test_bus_register_schema_and_validate(self):
        bus = AgentBus()
        bus.register_schema("task", TaskMessage)
        assert bus.schemas.get("task") is TaskMessage
        assert bus.validate_message("task", {"task_id": "x"}) == TaskMessage(task_id="x")

    def test_node_send_with_schema_validation(self):
        bus = AgentBus()
        bus.register_schema("task", TaskMessage)
        node = AgentNode("sender", handler=lambda m: "ok")
        bus.register(node)
        # Invalid content should raise before sending
        with pytest.raises(TypeError):
            node.send("someone", {"no_task_id": True}, schema="task")


# ── Dead Letter Queue (#2) ──────────────────────────────────────────────

class TestDeadLetterQueue:

    def test_record_and_inspect(self):
        dlq = DeadLetterQueue(max_retries=3)
        msg = AgentMessage(
            type=MessageType.REQUEST,
            sender=AgentAddress.local("a"),
            recipient=AgentAddress.local("b"),
            content="hi",
        )
        dlq.record(msg, "handler crashed", attempts=3)
        assert len(dlq) == 1
        assert dlq.messages[0].last_error == "handler crashed"

    def test_handler_callback(self):
        seen = []
        dlq = DeadLetterQueue(handler=lambda dl: seen.append(dl))
        msg = AgentMessage(type=MessageType.REQUEST, sender=AgentAddress.local("a"),
                           recipient=AgentAddress.local("b"), content="x")
        dlq.record(msg, "err", 1)
        assert len(seen) == 1

    def test_clear(self):
        dlq = DeadLetterQueue()
        msg = AgentMessage(type=MessageType.REQUEST, sender=AgentAddress.local("a"),
                           recipient=AgentAddress.local("b"), content="x")
        dlq.record(msg, "err", 1)
        dlq.clear()
        assert len(dlq) == 0

    def test_retry_unbound_raises(self):
        dlq = DeadLetterQueue()
        msg = AgentMessage(type=MessageType.REQUEST, sender=AgentAddress.local("a"),
                           recipient=AgentAddress.local("b"), content="x")
        dl = dlq.record(msg, "err", 1)
        with pytest.raises(RuntimeError):
            dlq.retry(dl)


class TestBusDLQIntegration:

    def test_error_reply_goes_to_dlq_after_retries(self):
        calls = {"n": 0}

        def failing_handler(message):
            calls["n"] += 1
            raise RuntimeError("boom")

        bus = AgentBus(dlq_enabled=True, max_retries=3)
        node = AgentNode("worker", handler=failing_handler)
        bus.register(node)

        reply = bus.send("client", "worker", "do it")
        # Retried 3 times, then dead-lettered
        assert calls["n"] == 3
        assert len(bus.dlq) == 1
        assert reply is not None and reply.type == MessageType.ERROR
        assert bus.stats["dead_lettered"] == 1

    def test_success_no_dlq(self):
        bus = AgentBus(dlq_enabled=True, max_retries=3)
        bus.register(AgentNode("worker", handler=lambda m: "fine"))
        reply = bus.send("client", "worker", "do it")
        assert reply.type == MessageType.REPLY
        assert len(bus.dlq) == 0

    def test_retry_recovers(self):
        calls = {"n": 0}

        def flaky_handler(message):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("temporary")
            return "recovered"

        bus = AgentBus(dlq_enabled=True, max_retries=3)
        bus.register(AgentNode("worker", handler=flaky_handler))
        reply = bus.send("client", "worker", "do it")
        assert reply.type == MessageType.REPLY
        assert reply.content == "recovered"
        assert len(bus.dlq) == 0


# ── Message Encryption (#5) ─────────────────────────────────────────────

@pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
class TestEncryption:

    def test_roundtrip(self):
        enc = MessageEncryptor(generate_key())
        original = AgentMessage(
            type=MessageType.REQUEST,
            sender=AgentAddress.local("a"),
            recipient=AgentAddress.local("b"),
            content={"secret": "trade secret"},
        )
        encrypted = enc.encrypt(original)
        assert encrypted.content != original.content
        assert encrypted.metadata.get("encrypted") is True

        decrypted = enc.decrypt(encrypted)
        assert decrypted.content == original.content

    def test_decrypt_non_encrypted_raises(self):
        enc = MessageEncryptor(generate_key())
        plain = AgentMessage(type=MessageType.REQUEST, sender=AgentAddress.local("a"),
                             recipient=AgentAddress.local("b"), content="plain")
        with pytest.raises(ValueError):
            enc.decrypt(plain)

    def test_missing_key_raises(self):
        import os
        old = os.environ.pop("AGENTLINK_ENCRYPTION_KEY", None)
        try:
            with pytest.raises(ValueError):
                MessageEncryptor()
        finally:
            if old:
                os.environ["AGENTLINK_ENCRYPTION_KEY"] = old

    def test_env_key(self, monkeypatch):
        monkeypatch.setenv("AGENTLINK_ENCRYPTION_KEY", generate_key())
        enc = MessageEncryptor()  # reads env var
        original = AgentMessage(type=MessageType.REQUEST, sender=AgentAddress.local("a"),
                                recipient=AgentAddress.local("b"), content="secret")
        assert enc.decrypt(enc.encrypt(original)).content == "secret"


# ── Protocol Gateway (#6) ───────────────────────────────────────────────

class TestProtocolGateway:

    def test_register_and_protocols(self):
        gw = ProtocolGateway(protocols=["agentlink", "a2a"])
        assert set(gw.protocols()) == {"agentlink", "a2a"}

    @pytest.mark.asyncio
    async def test_send_through_adapter(self):
        class FakeAdapter:
            def send(self, target, content, **kwargs):
                return f"sent to {target}: {content}"

        gw = ProtocolGateway()
        gw.register("a2a", FakeAdapter())
        result = await gw.send("a2a", "agent-x", "hello")
        assert result == "sent to agent-x: hello"

    @pytest.mark.asyncio
    async def test_send_unknown_protocol_raises(self):
        gw = ProtocolGateway()
        with pytest.raises(ValueError):
            await gw.send("nope", "x", "y")

    @pytest.mark.asyncio
    async def test_send_native_protocol_raises(self):
        gw = ProtocolGateway(protocols=["native"])
        with pytest.raises(RuntimeError):
            await gw.send("native", "x", "y")

    def test_on_handler(self):
        gw = ProtocolGateway()
        received = []

        @gw.on("autogen")
        def handle(msg):
            received.append(msg)

        gw.receive("autogen", "hello")
        assert received == ["hello"]


# ── WebSocket transport serialization (#4) ─────────────────────────────

class TestWSMessageSerialization:

    def test_serialize_roundtrip(self):
        msg = AgentMessage(
            type=MessageType.REQUEST,
            sender=AgentAddress.local("a"),
            recipient=AgentAddress.local("b"),
            content={"k": "v"},
        )
        data = serialize_message(msg)
        restored = deserialize_message(data)
        assert restored.id == msg.id
        assert restored.content == msg.content
        assert restored.sender.agent_id == "a"


@pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets not installed")
@pytest.mark.asyncio
class TestWSTransport:

    async def test_client_server_roundtrip(self):
        from agentlink.transport import WSTransport

        def handle(msg):
            return AgentMessage(
                type=MessageType.REPLY,
                sender=msg.recipient,
                recipient=msg.sender,
                content=f"echo: {msg.content}",
            )

        server = WSTransport(host="127.0.0.1", port=0, on_message=handle)
        await server.start()
        try:
            port = server._server.sockets[0].getsockname()[1]
            client = WSTransport()
            await client.connect(f"ws://127.0.0.1:{port}")
            try:
                reply = await client.send(AgentMessage(
                    type=MessageType.REQUEST,
                    sender=AgentAddress.local("a"),
                    recipient=AgentAddress.local("b"),
                    content="ping",
                ))
                assert reply is not None
                assert reply.content == "echo: ping"
            finally:
                await client.stop()
        finally:
            await server.stop()
