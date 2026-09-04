"""
Tests for the AgentLink Hub — distributed registry (v0.6.0).
"""

import time

import pytest

from agentlink.hub import HubClient, HubRegistry, HubServer


class TestHubRegistry:

    def test_register_and_get(self):
        registry = HubRegistry()
        reg = registry.register("researcher", capabilities=["web-search"], endpoint="http://r:1")
        assert reg.token
        fetched = registry.get("researcher")
        assert fetched is not None
        assert fetched["capabilities"] == ["web-search"]
        assert "token" not in fetched  # secret not exposed

    def test_discover_by_capability(self):
        registry = HubRegistry()
        registry.register("a", capabilities=["search"])
        registry.register("b", capabilities=["write"])
        hits = registry.discover(capabilities=["search"])
        assert [h["agent_id"] for h in hits] == ["a"]
        assert registry.discover(capabilities=["missing"]) == []

    def test_discover_by_namespace(self):
        registry = HubRegistry()
        registry.register("a", namespace="prod", capabilities=["x"])
        registry.register("b", namespace="dev", capabilities=["x"])
        hits = registry.discover(capabilities=["x"], namespace="prod")
        assert [h["agent_id"] for h in hits] == ["a"]

    def test_heartbeat_and_token_check(self):
        registry = HubRegistry()
        reg = registry.register("a")
        assert registry.heartbeat("a", reg.token) is True
        assert registry.heartbeat("a", "wrong-token") is False
        assert registry.heartbeat("ghost", reg.token) is False

    def test_expiry(self):
        registry = HubRegistry(ttl_seconds=0.2)
        registry.register("a")
        assert len(registry) == 1
        time.sleep(0.4)
        assert len(registry) == 0  # purged on access

    def test_re_register_keeps_original_registered_at(self):
        registry = HubRegistry()
        first = registry.register("a")
        time.sleep(0.05)
        second = registry.register("a")
        assert second.registered_at == first.registered_at
        assert second.token != first.token


class TestHubServerAndClient:
    def test_register_discover_deregister_roundtrip(self):
        with HubServer(ttl_seconds=30) as hub:
            client = HubClient(
                hub_url=hub.url,
                agent_id="researcher",
                capabilities=["web-search"],
                endpoint="http://me:1",
                heartbeat_interval=False,  # manual control
            ).register()
            try:
                peers = client.discover(capabilities=["web-search"])
                assert len(peers) == 1
                assert peers[0]["agent_id"] == "researcher"

                # other agents can see it too
                other = HubClient(hub_url=hub.url, agent_id="writer", heartbeat_interval=False).register()
                try:
                    assert other.discover(capabilities=["web-search"])[0]["endpoint"] == "http://me:1"
                finally:
                    other.deregister()
            finally:
                client.deregister()

            assert client.discover() == []

    def test_heartbeat_renews(self):
        with HubServer(ttl_seconds=0.3) as hub:
            client = HubClient(hub_url=hub.url, agent_id="a", heartbeat_interval=False).register()
            time.sleep(0.4)  # would expire
            assert hub.registry.get("a") is None
            client.register()  # re-register after expiry
            assert client.heartbeat() is True
            assert hub.registry.get("a") is not None

    def test_unknown_agent_heartbeat_404(self):
        with HubServer(ttl_seconds=30) as hub:
            client = HubClient(hub_url=hub.url, agent_id="ghost", heartbeat_interval=False)
            result = client._post("/heartbeat/ghost")
            assert result["ok"] is False

    def test_background_heartbeat_keeps_alive(self):
        with HubServer(ttl_seconds=0.5) as hub:
            # heartbeat every 0.15s keeps the registration alive past the TTL
            client = HubClient(hub_url=hub.url, agent_id="a", ttl_seconds=0.5, heartbeat_interval=0.15).start()
            try:
                time.sleep(1.2)  # > 2 TTLs
                assert hub.registry.get("a") is not None
            finally:
                client.stop()
            # after stop, no more heartbeats → expires
            time.sleep(0.7)
            assert hub.registry.get("a") is None

    def test_multiple_registrations_listing(self):
        with HubServer(ttl_seconds=30) as hub:
            c1 = HubClient(hub_url=hub.url, agent_id="a", capabilities=["x"], heartbeat_interval=False).register()
            c2 = HubClient(hub_url=hub.url, agent_id="b", capabilities=["y"], heartbeat_interval=False).register()
            try:
                agents = HubClient(hub_url=hub.url, agent_id="observer").agents()
                assert {a["agent_id"] for a in agents} == {"a", "b"}
            finally:
                c1.deregister()
                c2.deregister()
