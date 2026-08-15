"""
Dead Letter Queue (DLQ) for failed inter-agent messages (Issue #2).

When an agent fails to process a message (handler error, timeout, or no
recipient), the message is normally lost. The DLQ captures failed messages
after a configurable number of retries, so nothing is silently dropped.

Usage::

    from agentlink import AgentBus
    from agentlink.dlq import DeadLetterQueue

    def alert_ops(dead_letter):
        print(f"FAILED: {dead_letter.message.id} — {dead_letter.last_error}")

    bus = AgentBus(dlq_enabled=True, max_retries=3, dlq_handler=alert_ops)

    # ... later ...
    for failed in bus.dlq.messages:
        print(failed.last_error)
        bus.dlq.retry(failed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

from agentlink.protocol.message import AgentMessage


@dataclass
class DeadLetter:
    """A single failed message captured by the DLQ."""
    message: AgentMessage
    last_error: str
    attempts: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"DeadLetter(id={self.message.id!r}, attempts={self.attempts}, "
            f"error={self.last_error!r})"
        )


class DeadLetterQueue:
    """
    Collects messages that failed delivery after exhausting retries.

    Args:
        max_retries: Number of delivery attempts before dead-lettering.
        handler: Optional callable ``handler(dead_letter)`` invoked on capture.
    """

    def __init__(
        self,
        max_retries: int = 3,
        handler: Optional[Callable[[DeadLetter], Any]] = None,
    ) -> None:
        self.max_retries = max_retries
        self.handler = handler
        self._messages: List[DeadLetter] = []
        self._retry_fn: Optional[Callable[[DeadLetter], Optional[AgentMessage]]] = None

    # ── Internal binding ────────────────────────────────────────────────────

    def _bind_retry(self, retry_fn: Callable[[DeadLetter], Optional[AgentMessage]]) -> None:
        """Bind the retry callback (set by AgentBus)."""
        self._retry_fn = retry_fn

    # ── Capture ─────────────────────────────────────────────────────────────

    def record(
        self,
        message: AgentMessage,
        error: str,
        attempts: int,
        metadata: Optional[dict] = None,
    ) -> DeadLetter:
        """
        Record a failed message.

        Args:
            message: The message that failed.
            error: The last error message.
            attempts: Number of attempts that were made.
            metadata: Optional extra context.

        Returns:
            The created DeadLetter entry.
        """
        dl = DeadLetter(
            message=message,
            last_error=error,
            attempts=attempts,
            metadata=metadata or {},
        )
        self._messages.append(dl)
        if self.handler:
            self.handler(dl)
        return dl

    # ── Inspection ──────────────────────────────────────────────────────────

    @property
    def messages(self) -> List[DeadLetter]:
        """Return a copy of all dead-lettered messages."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    def clear(self) -> None:
        """Remove all dead-lettered messages."""
        self._messages.clear()

    # ── Recovery ────────────────────────────────────────────────────────────

    def retry(self, dead_letter: DeadLetter) -> Optional[AgentMessage]:
        """
        Manually retry a dead-lettered message.

        Args:
            dead_letter: The DeadLetter entry to retry.

        Returns:
            The reply message, or None if no retry callback is bound.

        Raises:
            RuntimeError: If the DLQ is not bound to a bus.
        """
        if self._retry_fn is None:
            raise RuntimeError("DLQ is not bound to an AgentBus")
        reply = self._retry_fn(dead_letter)
        if reply is not None:
            self._messages = [m for m in self._messages if m is not dead_letter]
        return reply

    def retry_all(self) -> List[Optional[AgentMessage]]:
        """Retry all dead-lettered messages. Returns the list of replies."""
        results = []
        for dl in list(self._messages):
            results.append(self.retry(dl))
        return results
