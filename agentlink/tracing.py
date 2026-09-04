"""
OpenTelemetry-compatible tracing for AgentLink (Roadmap).

Instruments an ``AgentBus`` so every routed message produces a span covering
delivery (and the reply, for request/reply flows). Works in two modes:

1. **Built-in recorder (zero dependencies)** — spans are collected in memory
   and inspectable via ``exporter.spans``. Ideal for tests and dashboards.
2. **OpenTelemetry** — pass an OTel tracer (``opentelemetry.trace.Tracer``)
   and spans are exported through your configured OTel pipeline.

Usage::

    from agentlink.tracing import instrument_bus, InMemorySpanExporter
    from agentlink import AgentBus

    exporter = InMemorySpanExporter()
    bus = instrument_bus(AgentBus(), service_name="my-fleet", exporter=exporter)

    bus.send("planner", "worker", "do it")
    for span in exporter.spans:
        print(span.name, span.duration_ms, span.status)

    # With OpenTelemetry installed:
    # from opentelemetry import trace
    # instrument_bus(bus, tracer=trace.get_tracer("agentlink"))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from agentlink.protocol.message import AgentMessage


@dataclass
class SpanRecord:
    """One recorded routing span (built-in recorder mode)."""
    name: str
    start_ns: int
    duration_ms: float
    attributes: dict = field(default_factory=dict)
    status: str = "ok"          # "ok" | "error"
    error: Optional[str] = None


class InMemorySpanExporter:
    """Collects finished spans in memory for inspection."""

    def __init__(self) -> None:
        self.spans: List[SpanRecord] = []

    def start_span(self, name: str, attributes: dict) -> int:
        return time.perf_counter_ns()

    def end_span(
        self,
        start_ns: int,
        name: str,
        attributes: dict,
        status: str = "ok",
        error: Optional[str] = None,
    ) -> SpanRecord:
        duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        record = SpanRecord(
            name=name,
            start_ns=start_ns,
            duration_ms=round(duration_ms, 3),
            attributes=attributes,
            status=status,
            error=error,
        )
        self.spans.append(record)
        return record

    def get_spans(self) -> List[SpanRecord]:
        return list(self.spans)

    def clear(self) -> None:
        self.spans.clear()


def _span_attributes(message: AgentMessage, service_name: str) -> dict:
    return {
        "service.name": service_name,
        "messaging.system": "agentlink",
        "message.id": message.id,
        "message.type": message.type.value,
        "message.sender": str(message.sender),
        "message.recipient": str(message.recipient),
        "correlation.id": message.correlation_id or "",
    }


def instrument_bus(
    bus: Any,
    service_name: str = "agentlink",
    exporter: Optional[InMemorySpanExporter] = None,
    tracer: Optional[Any] = None,
) -> Any:
    """
    Wrap ``bus._route`` so every routed message produces a tracing span.

    Args:
        bus: The AgentBus to instrument.
        service_name: Value of the ``service.name`` span attribute.
        exporter: An ``InMemorySpanExporter`` collecting spans in memory.
                  Ignored when ``tracer`` is provided.
        tracer: An OpenTelemetry ``trace.Tracer`` — when given, spans are
                created through OTel instead of the built-in recorder.

    Returns:
        The same bus, with ``_route`` wrapped (chainable).

    Raises:
        ValueError: If neither exporter nor tracer is provided.
    """
    if exporter is None and tracer is None:
        raise ValueError("instrument_bus requires an exporter or an OTel tracer")

    original_route: Callable = bus._route

    if tracer is not None:
        def traced_route_otel(message: AgentMessage, sender_node: Any = None, timeout: float = 30.0):
            attributes = _span_attributes(message, service_name)
            with tracer.start_as_current_span(
                f"agentlink.route.{message.type.value}", attributes=attributes
            ):
                return original_route(message, sender_node=sender_node, timeout=timeout)
        bus._route = traced_route_otel
        return bus

    assert exporter is not None

    def traced_route_builtin(message: AgentMessage, sender_node: Any = None, timeout: float = 30.0):
        attributes = _span_attributes(message, service_name)
        name = f"agentlink.route.{message.type.value}"
        start = exporter.start_span(name, attributes)
        try:
            reply = original_route(message, sender_node=sender_node, timeout=timeout)
            exporter.end_span(start, name, attributes, status="ok")
            return reply
        except Exception as e:  # noqa: BLE001 - record then re-raise
            exporter.end_span(start, name, attributes, status="error", error=str(e))
            raise

    bus._route = traced_route_builtin
    return bus
