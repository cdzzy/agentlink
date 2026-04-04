"""
Tests for the AgentLink protocol and runtime.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agentlink.protocol.message import AgentAddress, AgentMessage, MessageEnvelope, MessageType
from agentlink.protocol.capability import AgentCapability, CapabilitySet
from agentlink.runtime.node import AgentNode
from agentlink.runtime.bus import AgentBus, DeliveryError
from agentlink.runtime.registry import AgentRegistry
from agentlink.adapters.generic import GenericAdapter


# ── AgentAddress ─────────────────────────────────────────────────

def test_address_str():
    addr = AgentAddress("planner", "production")
    assert str(addr) == "planner@production"

def test_address_parse():
    addr = AgentAddress.parse("planner@production")
    assert addr.agent_id == "planner"
    assert addr.namespace == "production"

def test_address_parse_with_capability():
    addr = AgentAddress.parse("searcher@default/web-search")
    assert addr.agent_id == "searcher"
    assert addr.capability == "web-search"

def test_address_matches_wildcard():
    wildcard = AgentAddress("*", "*")
    target = AgentAddress("planner", "production")
    assert wildcard.matches(target)

def test_address_local():
    addr = AgentAddress.local("agent-x")
    assert addr.namespace == "default"


# ── AgentMessage ──────────────────────────────────────────────────

def test_message_creation():
    msg = AgentMessage(
        type=MessageType.REQUEST,
        sender=AgentAddress.local("a"),
        recipient=AgentAddress.local("b"),
        content="Hello!",
    )
    assert msg.id is not None
    assert msg.type == MessageType.REQUEST
    assert msg.content == "Hello!"

def test_message_reply():
    msg = AgentMessage(
        type=MessageType.REQUEST,
        sender=AgentAddress.local("a"),
        recipient=AgentAddress.local("b"),
        content="Question?",
    )
    reply = msg.reply("Answer!")
    assert reply.type == MessageType.REPLY
    assert reply.sender.agent_id == "b"
    assert reply.recipient.agent_id == "a"
    assert reply.correlation_id == msg.id
    assert reply.content == "Answer!"

def test_message_error():
    msg = AgentMessage(
        type=MessageType.REQUEST,
        sender=AgentAddress.local("a"),
        recipient=AgentAddress.local("b"),
        content="Do something",
    )
    err = msg.error("Something went wrong")
    assert err.type == MessageType.ERROR
    assert "Something went wrong" in err.content["error"]

def test_message_serialization():
    msg = AgentMessage(
        type=MessageType.REQUEST,
        sender=AgentAddress.local("sender"),
        recipient=AgentAddress.local("recipient"),
        content="test content",
    )
    d = msg.to_dict()
    restored = AgentMessage.from_dict(d)
    assert restored.id == msg.id
    assert restored.type == msg.type
    assert restored.content == msg.content
    assert str(restored.sender) == str(msg.sender)

def test_message_forward():
    msg = AgentMessage(
        type=MessageType.REQUEST,
        sender=AgentAddress.local("a"),
        recipient=AgentAddress.local("b"),
        content="Forward this",
    )
    fwd = msg.forward_to(AgentAddress.local("c"))
    assert fwd.recipient.agent_id == "c"
    assert fwd.type == MessageType.FORWARD
    assert "_forwarded_from" in fwd.metadata

def test_message_ttl_not_expired():
    msg = AgentMessage(
        type=MessageType.REQUEST,
        sender=AgentAddress.local("a"),
        recipient=AgentAddress.local("b"),
        content="x",
        ttl=3600,
    )
    assert not msg.is_expired()

def test_message_no_ttl_never_expired():
    msg = AgentMessage(
        type=MessageType.REQUEST,
        sender=AgentAddress.local("a"),
        recipient=AgentAddress.local("b"),
        content="x",
    )
    assert not msg.is_expired()


# ── MessageEnvelope ───────────────────────────────────────────────

def test_envelope_hop_tracking():
    msg = AgentMessage(
        type=MessageType.REQUEST,
        sender=AgentAddress.local("a"),
        recipient=AgentAddress.local("b"),
        content="x",
    )
    env = MessageEnvelope(message=msg)
    env.record_hop("node-1")
    env.record_hop("node-2")
    assert env.hop_count == 2
    assert "node-1" in env.route

def test_envelope_loop_detection():
    msg = AgentMessage(
        type=MessageType.REQUEST,
        sender=AgentAddress.local("a"),
        recipient=AgentAddress.local("b"),
        content="x",
    )
    env = MessageEnvelope(message=msg, max_hops=3)
    for i in range(3):
        env.record_hop(f"node-{i}")
    assert env.is_loop_detected()

def test_envelope_serialization():
    msg = AgentMessage(
        type=MessageType.REQUEST,
        sender=AgentAddress.local("a"),
        recipient=AgentAddress.local("b"),
        content="test",
    )
    env = MessageEnvelope(message=msg)
    d = env.to_dict()
    restored = MessageEnvelope.from_dict(d)
    assert restored.message.id == msg.id
    assert restored.protocol_version == env.protocol_version


# ── CapabilitySet ─────────────────────────────────────────────────

def test_capability_set_add_and_has():
    caps = CapabilitySet()
    caps.add(AgentCapability("web-search", tags=["search"]))
    assert caps.has("web-search")
    assert not caps.has("translate")

def test_capability_set_find_by_tag():
    caps = CapabilitySet([AgentCapability("web-search", tags=["search", "internet"])])
    found = caps.find("search")
    assert found is not None
    assert found.name == "web-search"

def test_capability_set_names():
    caps = CapabilitySet([
        AgentCapability("a"),
        AgentCapability("b"),
    ])
    assert set(caps.names()) == {"a", "b"}


# ── AgentRegistry ──────────────────────────────────────────────────

def make_node(agent_id, namespace="default", capabilities=None):
    return AgentNode(
        agent_id=agent_id,
        handler=lambda msg: "ok",
        namespace=namespace,
        capabilities=capabilities or [],
    )

def test_registry_register_and_find():
    reg = AgentRegistry()
    node = make_node("planner")
    reg.register(node)
    found = reg.find("planner")
    assert found is node

def test_registry_find_by_capability():
    reg = AgentRegistry()
    node = make_node("searcher", capabilities=["web-search"])
    reg.register(node)
    result = reg.find_by_capability("web-search")
    assert node in result

def test_registry_find_by_namespace():
    reg = AgentRegistry()
    n1 = make_node("a", namespace="prod")
    n2 = make_node("b", namespace="dev")
    reg.register(n1)
    reg.register(n2)
    prod = reg.find_by_namespace("prod")
    assert n1 in prod
    assert n2 not in prod

def test_registry_deregister():
    reg = AgentRegistry()
    node = make_node("temp")
    reg.register(node)
    reg.deregister("temp")
    assert reg.find("temp") is None

def test_registry_summary():
    reg = AgentRegistry()
    reg.register(make_node("a", capabilities=["search"]))
    reg.register(make_node("b", capabilities=["search", "summarize"]))
    summary = reg.summary()
    assert summary["total_agents"] == 2
    assert summary["capabilities"]["search"] == 2


# ── AgentBus ──────────────────────────────────────────────────────

def make_bus_with_nodes():
    bus = AgentBus("test-bus")
    n1 = make_node("alpha", capabilities=["search"])
    n2 = make_node("beta",  capabilities=["summarize"])
    bus.register(n1)
    bus.register(n2)
    return bus, n1, n2

def test_bus_direct_send():
    bus, n1, n2 = make_bus_with_nodes()
    reply = bus.send("alpha", "beta", "Hello beta!")
    assert reply is not None
    assert reply.type == MessageType.REPLY

def test_bus_send_not_found_raises():
    bus = AgentBus()
    n = make_node("a")
    bus.register(n)
    with pytest.raises(DeliveryError):
        bus.send("a", "nonexistent", "test")

def test_bus_node_to_node_send():
    bus, n1, n2 = make_bus_with_nodes()
    reply = n1.send("beta", "Hey beta!")
    assert reply is not None

def test_bus_capability_routing():
    bus = AgentBus()
    searcher = make_node("my-searcher", capabilities=["web-search"])
    client = make_node("client")
    bus.register_many(searcher, client)
    # Route to "web-search" capability — should find my-searcher
    reply = client.send("web-search", "Find something")
    assert reply is not None

def test_bus_ping_pong():
    bus, n1, n2 = make_bus_with_nodes()
    alive = n1.ping("beta")
    assert alive is True

def test_bus_ping_unknown_agent():
    bus, n1, n2 = make_bus_with_nodes()
    # ping catches DeliveryError internally and returns False
    alive = n1.ping("ghost-agent")
    assert alive is False

def test_bus_middleware_runs():
    bus = AgentBus()
    n1 = make_node("sender")
    n2 = make_node("receiver")
    bus.register_many(n1, n2)

    seen = []
    def mw(msg: AgentMessage) -> AgentMessage:
        seen.append(msg.id)
        return msg
    bus.use(mw)

    n1.send("receiver", "test")
    assert len(seen) > 0

def test_bus_middleware_can_block():
    bus = AgentBus()
    n1 = make_node("sender")
    n2 = make_node("receiver")
    bus.register_many(n1, n2)

    bus.use(lambda msg: None)   # drop everything

    # Blocked by middleware — bus routes 0 messages through to delivery
    # (returns None since middleware returned None)
    result = n1.send("receiver", "blocked")
    assert result is None

def test_bus_stats():
    bus, n1, n2 = make_bus_with_nodes()
    bus.send("alpha", "beta", "test1")
    bus.send("alpha", "beta", "test2")
    assert bus.stats["messages_routed"] >= 2

def test_bus_message_log():
    bus, n1, n2 = make_bus_with_nodes()
    bus.send("alpha", "beta", "log-this")
    assert len(bus.message_log) > 0


# ── GenericAdapter ────────────────────────────────────────────────

def test_generic_adapter_string_input():
    results = []

    def my_fn(text: str) -> str:
        results.append(text)
        return f"got: {text}"

    adapter = GenericAdapter(fn=my_fn, agent_id="test-agent")
    node = adapter.as_node()
    assert node.agent_id == "test-agent"

def test_generic_adapter_message_input():
    received = []

    def my_fn(msg: AgentMessage) -> str:
        # Only count REQUEST messages, ignore ANNOUNCE
        if msg.type == MessageType.REQUEST:
            received.append(msg)
        return "handled"

    adapter = GenericAdapter(fn=my_fn, agent_id="msg-agent", accept_message=True)
    node = adapter.as_node()

    bus = AgentBus()
    client = make_node("client")
    bus.register_many(node, client)

    reply = client.send("msg-agent", "test content")
    assert reply.content == "handled"
    assert len(received) == 1
    assert isinstance(received[0], AgentMessage)
    assert received[0].content == "test content"
