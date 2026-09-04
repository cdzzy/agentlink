"""
Tests for OpenTelemetry-compatible tracing (v0.4.0).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agentlink.runtime.node import AgentNode
from agentlink.runtime.bus import AgentBus, DeliveryError
from agentlink.tracing import instrument_bus, InMemorySpanExporter


def make_bus():
    bus = AgentBus()
    bus.register(AgentNode("worker", handler=lambda m: "done"))
    return bus


class TestInstrumentBus:

    def test_requires_exporter_or_tracer(self):
        with pytest.raises(ValueError):
            instrument_bus(AgentBus())

    def test_spans_recorded_for_routed_messages(self):
        exporter = InMemorySpanExporter()
        bus = instrument_bus(make_bus(), service_name="test-fleet", exporter=exporter)

        bus.send("client", "worker", "do it")

        spans = exporter.get_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "agentlink.route.request"
        assert span.status == "ok"
        assert span.attributes["service.name"] == "test-fleet"
        assert span.attributes["message.type"] == "request"
        assert span.attributes["messaging.system"] == "agentlink"
        assert span.duration_ms >= 0

    def test_error_status_on_delivery_failure(self):
        exporter = InMemorySpanExporter()
        bus = instrument_bus(AgentBus(), exporter=exporter)

        with pytest.raises(DeliveryError):
            bus.send("client", "nonexistent", "hi")

        spans = exporter.get_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"
        assert "No agent found" in (spans[0].error or "")

    def test_span_per_message(self):
        exporter = InMemorySpanExporter()
        bus = instrument_bus(make_bus(), exporter=exporter)

        bus.send("client", "worker", "one")
        bus.send("client", "worker", "two")

        assert len(exporter.get_spans()) == 2

    def test_clear(self):
        exporter = InMemorySpanExporter()
        bus = instrument_bus(make_bus(), exporter=exporter)
        bus.send("client", "worker", "one")
        exporter.clear()
        assert exporter.get_spans() == []

    def test_bus_still_routes_normally(self):
        exporter = InMemorySpanExporter()
        bus = instrument_bus(make_bus(), exporter=exporter)

        reply = bus.send("client", "worker", "hello")
        assert reply is not None
        assert reply.content == "done"

    def test_otel_tracer_mode(self):
        """With an OTel-style tracer, spans flow through start_as_current_span."""

        class FakeSpan:
            def __init__(self):
                self.ended = False

        class FakeTracer:
            def __init__(self):
                self.spans = []

            def start_as_current_span(self, name, attributes=None):
                span = FakeSpan()
                self.spans.append((name, attributes, span))
                import contextlib

                @contextlib.contextmanager
                def ctx():
                    yield span
                    span.ended = True
                return ctx()

        tracer = FakeTracer()
        bus = instrument_bus(make_bus(), service_name="otel-fleet", tracer=tracer)

        bus.send("client", "worker", "do it")

        assert len(tracer.spans) == 1
        name, attributes, span = tracer.spans[0]
        assert name == "agentlink.route.request"
        assert attributes["service.name"] == "otel-fleet"
        assert span.ended is True

    def test_instrumenting_twice_chains(self):
        exporter1 = InMemorySpanExporter()
        exporter2 = InMemorySpanExporter()
        bus = instrument_bus(make_bus(), exporter=exporter1)
        instrument_bus(bus, exporter=exporter2)

        bus.send("client", "worker", "one")

        # Each instrumented layer records one span for the same route
        assert len(exporter1.get_spans()) == 1
        assert len(exporter2.get_spans()) == 1
