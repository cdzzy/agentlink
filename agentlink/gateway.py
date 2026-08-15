"""
Multi-protocol gateway for heterogeneous agent ecosystems (Issue #6).

The AI agent ecosystem is fragmented (LangChain, AutoGen, CrewAI, A2A each
use incompatible message formats). The ProtocolGateway is a translation layer
that lets AgentLink agents interoperate with agents built on other protocols.

Usage::

    from agentlink.gateway import ProtocolGateway

    gateway = ProtocolGateway(protocols=["agentlink", "a2a", "langchain"])

    # Register a protocol adapter (any object with a send() method)
    gateway.register("a2a", a2a_adapter)

    # Send through a specific protocol
    await gateway.send("a2a", "researcher@other-platform", {"task": "Analyze data"})

    # Receive from a protocol (auto-translated to AgentLink format)
    @gateway.on("autogen")
    def handle_autogen_message(msg):
        print(msg)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from agentlink.protocol.message import AgentMessage


class ProtocolGateway:
    """
    Routes messages across multiple agent protocols.

    Each protocol is backed by an adapter object exposing a ``send`` method
    (e.g. ``A2AAdapter``). Receive handlers can be registered per protocol
    via the ``on`` decorator.
    """

    def __init__(self, protocols: Optional[List[str]] = None) -> None:
        self._adapters: Dict[str, Any] = {}
        self._handlers: Dict[str, List[Callable[[AgentMessage], Any]]] = {}

        for protocol in (protocols or []):
            self.register_protocol(protocol)

    def register_protocol(self, protocol: str, adapter: Any = None) -> "ProtocolGateway":
        """
        Register a protocol name (and optionally an adapter).

        If no adapter is provided, the protocol is registered as "native"
        and messages are passed through unchanged.

        Args:
            protocol: Protocol name (e.g. ``"a2a"``, ``"langchain"``, ``"autogen"``).
            adapter: Object with a ``send(target, content, **kwargs)`` method.

        Returns:
            self (for chaining).
        """
        self._adapters[protocol] = adapter
        self._handlers.setdefault(protocol, [])
        return self

    def register(self, protocol: str, adapter: Any) -> "ProtocolGateway":
        """Alias for :meth:`register_protocol` with an adapter."""
        return self.register_protocol(protocol, adapter)

    def protocols(self) -> List[str]:
        """Return the list of registered protocol names."""
        return list(self._adapters.keys())

    def has(self, protocol: str) -> bool:
        return protocol in self._adapters

    async def send(self, protocol: str, target: str, content: Any, **kwargs: Any) -> Any:
        """
        Send a message through the given protocol.

        Args:
            protocol: The protocol to use.
            target: Target agent identifier.
            content: Message content.
            **kwargs: Passed through to the adapter's ``send``.

        Returns:
            The reply from the adapter.

        Raises:
            ValueError: If the protocol is not registered.
            RuntimeError: If the protocol has no adapter to send through.
        """
        if protocol not in self._adapters:
            raise ValueError(
                f"Unknown protocol {protocol!r}. Registered: {self.protocols()}"
            )
        adapter = self._adapters[protocol]
        if adapter is None:
            raise RuntimeError(
                f"Protocol {protocol!r} is registered as native and has no adapter to send through."
            )
        send = getattr(adapter, "send", None)
        if send is None:
            raise RuntimeError(f"Adapter for {protocol!r} has no send() method")
        result = send(target, content, **kwargs)
        if hasattr(result, "__await__"):
            return await result
        return result

    def on(self, protocol: str) -> Callable:
        """
        Decorator to register a receive handler for a protocol.

        Example::

            @gateway.on("autogen")
            def handle(msg):
                ...
        """

        def decorator(fn: Callable[[AgentMessage], Any]) -> Callable:
            self._handlers.setdefault(protocol, []).append(fn)
            return fn

        return decorator

    def receive(self, protocol: str, message: AgentMessage) -> None:
        """
        Dispatch an incoming message to the protocol's registered handlers.

        Args:
            protocol: The protocol the message arrived on.
            message: The (already-translated) AgentMessage.
        """
        for handler in self._handlers.get(protocol, []):
            handler(message)

    def handler_count(self, protocol: str) -> int:
        return len(self._handlers.get(protocol, []))
