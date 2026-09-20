"""
Tests for the A2A v1.0 compatibility layer:
- Agent Card generation/parsing (/.well-known/agent-card.json)
- AgentMessage ⇄ A2A Task/Artifact lossless mapping
- A2ATransport JSON-RPC client/server bridging
- Agent Card JWKS signature verification (requires the [a2a] extras)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import json

import pytest

from agentlink.a2a import (
    A2A_PROTOCOL_VERSION,
    AGENT_CARD_WELLKNOWN_PATH,
    ENVELOPE_METADATA_KEY,
    TASK_STATE_COMPLETED,
    TASK_STATE_SUBMITTED,
    A2ATransport,
    A2ATransportError,
    AgentCapabilities,
    AgentCard,
    AgentCardSignature,
    AgentSkill,
    a2a_message_to_agent_message,
    a2a_task_to_agent_message,
    agent_card_url,
    agent_message_to_a2a_message,
    agent_message_to_a2a_task,
    content_to_part,
    has_agentlink_envelope,
    is_terminal_state,
    parts_to_content,
    strip_envelope,
    task_reply_to_a2a_task,
)
from agentlink.protocol.message import AgentAddress, AgentMessage, MessageType

try:
    import cryptography  # noqa: F401
    import jwt  # noqa: F401

    from agentlink.a2a import (
        A2AVerificationError,
        canonical_card_bytes,
        generate_signing_key,
        sign_agent_card,
        verify_agent_card,
    )
    HAS_A2A_CRYPTO = True
except ImportError:
    HAS_A2A_CRYPTO = False


def make_message(**overrides) -> AgentMessage:
    """Build a fully-populated AgentMessage for lossless round-trip checks."""
    defaults = dict(
        type=MessageType.REQUEST,
        sender=AgentAddress("planner", "my-app", capability="web-search"),
        recipient=AgentAddress("researcher", "my-app"),
        content="Analyze Q1 market data",
        correlation_id="corr-123",
        parent_id="parent-456",
        ttl=60,
        metadata={"user_tag": "u1", "nested": {"k": [1, 2, {"deep": True}]}},
        content_type="text/plain",
    )
    defaults.update(overrides)
    return AgentMessage(**defaults)


def make_card() -> AgentCard:
    """Build a representative v1.0 Agent Card."""
    card = AgentCard(
        name="market-researcher",
        description="Research agent for market data analysis",
        version="0.6.0",
        skills=[
            AgentSkill(
                id="web-search",
                name="Web Search",
                description="Searches the web",
                tags=["search", "web"],
                examples=["Find AI news", "Market summary"],
                inputModes=["text", "application/json"],
                outputModes=["text"],
            ),
            AgentSkill(id="summarize", name="Summarize", tags=["nlp"]),
        ],
        provider={"organization": "cdzzy", "url": "https://github.com/cdzzy/agentlink"},
        documentationUrl="https://github.com/cdzzy/agentlink#readme",
        capabilities=AgentCapabilities(streaming=True, pushNotifications=False),
    )
    card.url = "https://agent.example.com/a2a"
    return card


# ── Agent Card: generation & parsing ─────────────────────────────────────────

class TestAgentCard:

    def test_wellknown_path_constant(self):
        assert AGENT_CARD_WELLKNOWN_PATH == "/.well-known/agent-card.json"
        assert A2A_PROTOCOL_VERSION == "1.0"

    def test_agent_card_url(self):
        assert agent_card_url("https://agent.example.com") == \
            "https://agent.example.com/.well-known/agent-card.json"
        assert agent_card_url("https://agent.example.com/") == \
            "https://agent.example.com/.well-known/agent-card.json"

    def test_card_dict_roundtrip_lossless(self):
        card = make_card()
        parsed = AgentCard.from_dict(card.to_dict())
        assert parsed.name == card.name
        assert parsed.description == card.description
        assert parsed.version == card.version
        assert parsed.url == "https://agent.example.com/a2a"
        assert len(parsed.skills) == 2
        assert parsed.skills[0].id == "web-search"
        assert parsed.skills[0].tags == ["search", "web"]
        assert parsed.skills[0].examples == ["Find AI news", "Market summary"]
        assert parsed.capabilities.streaming is True
        assert parsed.capabilities.pushNotifications is False
        assert parsed.provider == card.provider
        assert parsed.documentationUrl == card.documentationUrl
        assert parsed.supportedInterfaces[0].protocolBinding == "JSONRPC"
        assert parsed.supportedInterfaces[0].protocolVersion == "1.0"
        # Re-serialization is stable (idempotent round-trip).
        assert parsed.to_dict() == card.to_dict()

    def test_card_json_bytes_roundtrip(self):
        card = make_card()
        raw = card.to_json_bytes()
        assert b'"agent-card"' not in raw  # sanity: plain JSON payload
        parsed = AgentCard.from_bytes(raw)
        assert parsed.to_dict() == card.to_dict()
        assert json.loads(parsed.to_json()) == card.to_dict()

    def test_url_property_synthesizes_interface(self):
        card = AgentCard(name="bare")
        assert card.url == ""
        card.url = "https://x.example.com/a2a"
        assert card.url == "https://x.example.com/a2a"
        assert card.supportedInterfaces[0].protocolBinding == "JSONRPC"
        assert card.supportedInterfaces[0].protocolVersion == "1.0"

    def test_legacy_v03_card_normalized(self):
        """v0.3 flat card (top-level url) is normalized into supportedInterfaces."""
        legacy = {
            "name": "legacy-agent",
            "description": "old style card",
            "url": "https://old.example.com/a2a",
            "preferredTransport": "JSONRPC",
            "protocolVersion": "0.3.0",
            "skills": [{"id": "s1", "name": "Skill One"}],
        }
        card = AgentCard.from_dict(legacy)
        assert card.url == "https://old.example.com/a2a"
        assert card.supportedInterfaces[0].protocolVersion == "0.3.0"
        assert card.skills[0].name == "Skill One"

    def test_unknown_fields_preserved_in_extra(self):
        payload = make_card().to_dict()
        payload["x-custom-extension"] = {"enabled": True}
        card = AgentCard.from_dict(payload)
        assert card.extra["x-custom-extension"] == {"enabled": True}
        # and it survives re-serialization
        assert card.to_dict()["x-custom-extension"] == {"enabled": True}

    def test_signatures_parse_roundtrip(self):
        payload = make_card().to_dict()
        payload["signatures"] = [{
            "protected": "eyJhbGciOiJFUzI1NiJ9",
            "signature": "c2ln",
        }]
        card = AgentCard.from_dict(payload)
        assert len(card.signatures) == 1
        assert isinstance(card.signatures[0], AgentCardSignature)
        assert card.to_dict()["signatures"] == payload["signatures"]

    def test_exported_from_package_root(self):
        import agentlink
        assert agentlink.AgentCard is AgentCard
        assert agentlink.A2ATransport is A2ATransport
        assert agentlink.AGENT_CARD_WELLKNOWN_PATH == AGENT_CARD_WELLKNOWN_PATH


# ── Mapping: content ⇄ Part ──────────────────────────────────────────────────

class TestContentParts:

    def test_text_part(self):
        part = content_to_part("hello")
        assert part == {"text": "hello", "mediaType": "text/plain"}
        assert parts_to_content([part]) == "hello"

    def test_data_part(self):
        for payload in ({"a": 1}, [1, 2], 42, 3.14, True, None):
            part = content_to_part(payload)
            assert part["data"] == payload
            assert part["mediaType"] == "application/json"
            assert parts_to_content([part]) == payload

    def test_raw_part(self):
        blob = b"\x00\x01binary"
        part = content_to_part(blob)
        assert part["raw"] == base64.b64encode(blob).decode("ascii")
        assert part["mediaType"] == "application/octet-stream"
        assert parts_to_content([part]) == blob

    def test_unsupported_content_raises(self):
        with pytest.raises(ValueError):
            content_to_part(object())

    def test_multiple_text_parts_joined(self):
        assert parts_to_content([
            {"text": "a"}, {"text": "b"},
        ]) == "a\nb"

    def test_legacy_kind_part_tolerated(self):
        legacy_file = {
            "kind": "file",
            "file": {"fileWithUri": "https://f/x.pdf", "mimeType": "application/pdf"},
        }
        content = parts_to_content([legacy_file])
        assert content == {"url": "https://f/x.pdf", "mediaType": "application/pdf"}


# ── Mapping: AgentMessage ⇄ A2A Message/Task ─────────────────────────────────

class TestMessageMapping:

    def test_message_roundtrip_lossless(self):
        msg = make_message()
        wire = agent_message_to_a2a_message(msg)
        back = a2a_message_to_agent_message(wire)
        assert back.type == msg.type
        assert str(back.sender) == str(msg.sender)
        assert str(back.recipient) == str(msg.recipient)
        assert back.content == msg.content
        assert back.id == msg.id
        assert back.correlation_id == msg.correlation_id
        assert back.parent_id == msg.parent_id
        assert back.timestamp == msg.timestamp
        assert back.metadata == msg.metadata
        assert back.ttl == msg.ttl
        assert back.content_type == msg.content_type

    def test_wire_shape_v1_conventions(self):
        wire = agent_message_to_a2a_message(make_message())
        assert wire["messageId"]
        assert wire["role"] == "ROLE_USER"
        assert wire["taskId"] == wire["messageId"]
        assert wire["contextId"] == "corr-123"
        assert wire["parts"][0]["text"] == "Analyze Q1 market data"
        assert ENVELOPE_METADATA_KEY in wire["metadata"]

    def test_reply_maps_to_role_agent(self):
        wire = agent_message_to_a2a_message(
            make_message(type=MessageType.REPLY, content="ok"))
        assert wire["role"] == "ROLE_AGENT"

    def test_native_message_synthesized_without_envelope(self):
        foreign = {
            "messageId": "m-1",
            "role": "ROLE_AGENT",
            "parts": [{"text": "hi", "mediaType": "text/plain"}],
            "metadata": {"trace": "t-9"},
        }
        msg = a2a_message_to_agent_message(foreign)
        assert msg.type is MessageType.REPLY
        assert msg.content == "hi"
        assert msg.id == "m-1"
        assert msg.metadata == {"trace": "t-9"}
        assert msg.sender.agent_id == "a2a-client"
        assert msg.recipient.agent_id == "a2a-agent"

    def test_legacy_role_tolerated(self):
        msg = a2a_message_to_agent_message({
            "messageId": "m-2", "role": "agent",
            "parts": [{"text": "x"}],
        })
        assert msg.type is MessageType.REPLY
        msg = a2a_message_to_agent_message({
            "messageId": "m-3", "role": "user",
            "parts": [{"text": "y"}],
        })
        assert msg.type is MessageType.REQUEST

    def test_envelope_helpers(self):
        wire = agent_message_to_a2a_message(make_message())
        assert has_agentlink_envelope(wire["metadata"])
        stripped = strip_envelope(wire["metadata"])
        assert not has_agentlink_envelope(stripped)
        assert stripped == {"user_tag": "u1", "nested": {"k": [1, 2, {"deep": True}]}}


class TestTaskMapping:

    def test_task_roundtrip_lossless_send_receive(self):
        msg = make_message()
        task = agent_message_to_a2a_task(msg)
        back = a2a_task_to_agent_message(task)
        assert back.type == msg.type
        assert str(back.sender) == str(msg.sender)
        assert str(back.recipient) == str(msg.recipient)
        assert back.content == msg.content
        assert back.id == msg.id
        assert back.correlation_id == msg.correlation_id
        assert back.parent_id == msg.parent_id
        assert back.timestamp == msg.timestamp
        assert back.metadata == msg.metadata
        assert back.ttl == msg.ttl
        assert back.content_type == msg.content_type

    def test_task_ids_and_state(self):
        msg = make_message()
        task = agent_message_to_a2a_task(msg)
        assert task["id"] == msg.id
        assert task["contextId"] == "corr-123"
        assert task["status"]["state"] == TASK_STATE_SUBMITTED
        custom = agent_message_to_a2a_task(msg, state=TASK_STATE_COMPLETED)
        assert custom["status"]["state"] == TASK_STATE_COMPLETED

    def test_task_reply_roundtrip_with_artifact(self):
        task = agent_message_to_a2a_task(make_message())
        inbound = a2a_task_to_agent_message(task)
        reply = inbound.reply("result-42")
        completed = task_reply_to_a2a_task(task, reply, artifact_name="answer")

        # Native A2A consumers read the artifact...
        assert completed["status"]["state"] == TASK_STATE_COMPLETED
        assert completed["artifacts"][0]["name"] == "answer"
        assert completed["artifacts"][0]["parts"][0]["text"] == "result-42"
        assert completed["id"] == task["id"]

        # ...while AgentLink peers restore the reply losslessly.
        got = a2a_task_to_agent_message(completed, original=inbound)
        assert got.content == "result-42"
        assert got.type is MessageType.REPLY
        assert got.correlation_id == inbound.id
        assert got.parent_id == inbound.id
        assert str(got.sender) == str(inbound.recipient)

    def test_foreign_task_synthesized_from_artifacts(self):
        foreign = {
            "id": "task-9",
            "contextId": "ctx-9",
            "status": {"state": "TASK_STATE_COMPLETED", "timestamp": "2026-01-01T00:00:00.000Z"},
            "artifacts": [{"parts": [{"text": "artifact payload"}]}],
            "history": [],
        }
        msg = a2a_task_to_agent_message(foreign)
        assert msg.content == "artifact payload"
        assert msg.type is MessageType.REPLY
        assert msg.correlation_id == "ctx-9"

    def test_foreign_task_with_original_uses_reply_correlation(self):
        original = make_message()
        foreign = {
            "id": "task-x",
            "status": {"state": "completed"},
            "artifacts": [{"parts": [{"data": {"answer": 7}}]}],
        }
        msg = a2a_task_to_agent_message(foreign, original=original)
        assert msg.content == {"answer": 7}
        assert msg.correlation_id == original.id

    def test_terminal_states(self):
        assert is_terminal_state("TASK_STATE_COMPLETED")
        assert is_terminal_state("TASK_STATE_FAILED")
        assert is_terminal_state("TASK_STATE_CANCELED")
        assert is_terminal_state("TASK_STATE_REJECTED")
        # legacy lowercase tolerated
        assert is_terminal_state("completed")
        assert is_terminal_state("task-state-failed")
        assert not is_terminal_state("TASK_STATE_WORKING")
        assert not is_terminal_state("TASK_STATE_SUBMITTED")


# ── Transport: JSON-RPC bridging ─────────────────────────────────────────────

class TestA2ATransport:

    def test_build_request_shape(self):
        transport = A2ATransport(url="https://x.example.com/a2a/")
        assert transport.url == "https://x.example.com/a2a"
        request = transport.build_request(make_message(), request_id="req-1")
        assert request["jsonrpc"] == "2.0"
        assert request["id"] == "req-1"
        assert request["method"] == "SendMessage"
        assert request["params"]["message"]["role"] == "ROLE_USER"

    def test_parse_request_accepts_v1_and_legacy(self):
        transport = A2ATransport(local_agent_id="svc")
        payload = agent_message_to_a2a_message(make_message())
        for method in ("SendMessage", "message/send", "message:send"):
            inbound = transport.parse_request({
                "jsonrpc": "2.0", "id": 1, "method": method,
                "params": {"message": payload},
            })
            assert inbound.content == "Analyze Q1 market data"
            # envelope restore keeps the original AgentLink recipient
            assert inbound.recipient.agent_id == "researcher"
        with pytest.raises(A2ATransportError):
            transport.parse_request({"method": "tasks/get", "params": {}})
        with pytest.raises(A2ATransportError):
            transport.parse_request({"method": "SendMessage", "params": {}})

    def test_parse_request_foreign_message_routed_to_local_agent(self):
        transport = A2ATransport(local_agent_id="svc")
        foreign = {
            "messageId": "f-1",
            "role": "ROLE_USER",
            "parts": [{"text": "ping"}],
            "metadata": {},
        }
        inbound = transport.parse_request({
            "jsonrpc": "2.0", "id": 2, "method": "SendMessage",
            "params": {"message": foreign},
        })
        assert inbound.recipient.agent_id == "svc"
        assert inbound.sender.agent_id == "a2a-client"
        assert inbound.content == "ping"

    def test_full_server_client_roundtrip_lossless(self):
        """send → parse_request → reply → build_success_response → parse_response."""
        transport = A2ATransport(local_agent_id="svc")
        original = make_message()

        request = transport.build_request(original, request_id="rpc-7")
        inbound = transport.parse_request(request)
        # Lossless delivery of the request.
        assert inbound.id == original.id
        assert inbound.metadata == original.metadata
        assert inbound.ttl == original.ttl

        reply = inbound.reply("done")
        response = transport.build_success_response(
            reply, request_id=request["id"], original=inbound)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "rpc-7"
        assert "result" in response

        got = transport.parse_response(response["result"], original)
        assert got.content == "done"
        assert got.type is MessageType.REPLY
        assert got.correlation_id == original.id
        assert got.parent_id == reply.parent_id

    def test_parse_response_direct_message_coerced_to_reply(self):
        transport = A2ATransport()
        sent = make_message()
        result = {
            "messageId": "m-r",
            "role": "ROLE_USER",  # even a foreign ROLE_USER message is a reply to us
            "parts": [{"text": "quick answer"}],
            "metadata": {},
        }
        msg = transport.parse_response(result, sent)
        assert msg.type is MessageType.REPLY
        assert msg.content == "quick answer"
        assert msg.correlation_id == sent.id

    def test_build_error_response(self):
        err = A2ATransport.build_error_response(-32601, "Method not found", "id-9")
        assert err == {
            "jsonrpc": "2.0", "id": "id-9",
            "error": {"code": -32601, "message": "Method not found"},
        }

    def test_send_without_url_raises(self):
        with pytest.raises(A2ATransportError):
            A2ATransport().send(make_message())

    def test_from_card(self):
        card = make_card()
        transport = A2ATransport.from_card(card)
        assert transport.url == "https://agent.example.com/a2a"
        bare = AgentCard(name="no-interface")
        with pytest.raises(A2ATransportError):
            A2ATransport.from_card(bare)

    async def test_asend_delegates_to_send(self, monkeypatch):
        transport = A2ATransport(url="https://x.example.com/a2a")
        reply = make_message(type=MessageType.REPLY, content="async-ok")
        monkeypatch.setattr(transport, "send", lambda message: reply)
        assert await transport.asend(make_message()) is reply


# ── Verification: JWKS / detached JWS ────────────────────────────────────────

@pytest.mark.skipif(not HAS_A2A_CRYPTO, reason="pyjwt/cryptography not installed ([a2a] extras)")
class TestCardVerification:

    def test_es256_self_signed_roundtrip(self):
        private_pem, jwks = generate_signing_key("ES256", kid="k-es")
        card = make_card()
        signed = sign_agent_card(card, private_pem, kid="k-es")
        assert signed is card
        assert len(card.signatures) == 1
        assert verify_agent_card(card, jwks) is True
        # survives a JSON round-trip (simulating the HTTP response body)
        assert verify_agent_card(AgentCard.from_json(card.to_json()), jwks) is True

    def test_rs256_self_signed_roundtrip(self):
        private_pem, jwks = generate_signing_key("RS256", kid="k-rs")
        card = make_card()
        sign_agent_card(card, private_pem, kid="k-rs", algorithm="RS256")
        assert verify_agent_card(card, jwks) is True

    def test_eddsa_self_signed_roundtrip(self):
        private_pem, jwks = generate_signing_key("EdDSA", kid="k-ed")
        card = make_card()
        sign_agent_card(card, private_pem, kid="k-ed", algorithm="EdDSA")
        assert verify_agent_card(card, jwks) is True

    def test_tampered_name_detected(self):
        private_pem, jwks = generate_signing_key("ES256", kid="k1")
        card = make_card()
        sign_agent_card(card, private_pem, kid="k1")
        card.name = "evil-agent"
        with pytest.raises(A2AVerificationError):
            verify_agent_card(card, jwks)

    def test_tampered_skill_detected(self):
        private_pem, jwks = generate_signing_key("ES256", kid="k1")
        card = make_card()
        sign_agent_card(card, private_pem, kid="k1")
        card.skills.append(AgentSkill(id="backdoor", name="Backdoor"))
        with pytest.raises(A2AVerificationError):
            verify_agent_card(card, jwks)

    def test_unsigned_card_rejected(self):
        _, jwks = generate_signing_key("ES256", kid="k1")
        card = make_card()
        with pytest.raises(A2AVerificationError):
            verify_agent_card(card, jwks)
        # explicitly allowed when require_signature=False
        assert verify_agent_card(card, jwks, require_signature=False) is True

    def test_wrong_key_detected(self):
        private_pem, _ = generate_signing_key("ES256", kid="k1")
        _, attacker_jwks = generate_signing_key("ES256", kid="k2")
        card = make_card()
        sign_agent_card(card, private_pem, kid="k1")
        with pytest.raises(A2AVerificationError):
            verify_agent_card(card, attacker_jwks)

    def test_kid_mismatch_rejected(self):
        private_pem, jwks = generate_signing_key("ES256", kid="k1")
        card = make_card()
        sign_agent_card(card, private_pem, kid="k1")
        with pytest.raises(A2AVerificationError):
            verify_agent_card(card, jwks, kid="other-key")

    def test_no_kid_header_roundtrip(self):
        # README quick-start path: no explicit kid anywhere. The JWS header
        # carries no ``kid`` while the JWKS key keeps its generated one —
        # per RFC 7515 verification must still succeed.
        private_pem, jwks = generate_signing_key("ES256")
        card = make_card()
        sign_agent_card(card, private_pem)
        header = json.loads(
            base64.urlsafe_b64decode(card.signatures[0].protected + "==")
        )
        assert "kid" not in header
        assert verify_agent_card(card, jwks) is True

    def test_no_kid_header_picks_right_key_in_multi_key_jwks(self):
        _, jwks_a = generate_signing_key("ES256", kid="k-a")
        private_b, jwks_b = generate_signing_key("ES256", kid="k-b")
        combined = {"keys": jwks_a["keys"] + jwks_b["keys"]}
        card = make_card()
        sign_agent_card(card, private_b)  # header without kid
        assert verify_agent_card(card, combined) is True
        # ... but a set without the signing key still rejects
        with pytest.raises(A2AVerificationError):
            verify_agent_card(card, jwks_a)

    def test_signatures_excluded_from_signed_payload(self):
        private_pem, _ = generate_signing_key("ES256", kid="k1")
        card = make_card()
        unsigned_payload = canonical_card_bytes(card)
        sign_agent_card(card, private_pem, kid="k1")
        assert canonical_card_bytes(card) == unsigned_payload

    def test_dict_signing_path(self):
        private_pem, jwks = generate_signing_key("ES256", kid="k1")
        payload = make_card().to_dict()
        out = sign_agent_card(payload, private_pem, kid="k1")
        assert isinstance(out, dict)
        assert out["signatures"][0]["protected"]
        assert verify_agent_card(out, jwks) is True

    def test_unsupported_algorithm_rejected(self):
        private_pem, jwks = generate_signing_key("ES256", kid="k1")
        card = make_card()
        sign_agent_card(card, private_pem, kid="k1")
        forged = json.loads(card.to_json())
        forged["signatures"] = [{
            "protected": base64.urlsafe_b64encode(
                b'{"alg":"HS256","kid":"k1","typ":"JOSE"}').rstrip(b"=").decode(),
            "signature": "AAAA",
        }]
        with pytest.raises(A2AVerificationError):
            verify_agent_card(forged, jwks)

    def test_verify_package_root_export(self):
        import agentlink
        assert agentlink.verify_agent_card is verify_agent_card
        assert agentlink.A2AVerificationError is A2AVerificationError
