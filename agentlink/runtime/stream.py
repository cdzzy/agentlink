"""
Streaming support for long-running agent tasks (Roadmap).

Enables chunked, incremental responses between agents using the STREAM_START /
STREAM_CHUNK / STREAM_END message types.

Usage::

    # Receiver handler returns an iterable of chunks — the node streams them
    def summarize(message):
        for part in long_running_summarizer(message.content):
            yield part

    node = AgentNode("summarizer", handler=summarize)
    bus.register(node)

    # Sender consumes chunks as they arrive
    stream = client.stream("summarizer", "Summarize this long document")
    for chunk in stream:
        print(chunk, end="")
    print(stream.collect())   # full text

The stream protocol uses the request's correlation id so multiple concurrent
streams never interleave.
"""

from __future__ import annotations

import threading
from typing import Iterable, Iterator, Optional

from agentlink.protocol.message import AgentMessage, MessageType


class StreamResult:
    """
    Collects streamed chunks for one request.

    Chunks are appended as STREAM_CHUNK messages arrive; iteration yields them
    as they come in and stops after STREAM_END.
    """

    def __init__(self, correlation_id: str, timeout: float = 60.0) -> None:
        self.correlation_id = correlation_id
        self.timeout = timeout
        self._chunks: list = []
        self._finished = False
        self._failed: Optional[str] = None
        self._event = threading.Event()

    def push(self, chunk: object) -> None:
        """Append a chunk (called by the receiving node)."""
        self._chunks.append(chunk)
        self._event.set()

    def finish(self) -> None:
        """Mark the stream complete."""
        self._finished = True
        self._event.set()

    def fail(self, error: str) -> None:
        """Mark the stream failed."""
        self._failed = error
        self._finished = True
        self._event.set()

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def failed(self) -> Optional[str]:
        return self._failed

    def __iter__(self) -> Iterator[object]:
        """Yield chunks as they arrive, stopping at STREAM_END."""
        index = 0
        while True:
            while index < len(self._chunks):
                yield self._chunks[index]
                index += 1
            if self._finished:
                break
            if not self._event.wait(self.timeout):
                raise TimeoutError(f"Stream {self.correlation_id} timed out after {self.timeout}s")
            self._event.clear()
        if self._failed:
            raise RuntimeError(self._failed)

    def collect(self) -> str:
        """Block until the stream completes and return the joined content."""
        for _ in self:  # noqa: B007 - drive iteration to completion
            pass
        return "".join(str(c) for c in self._chunks)

    def chunks(self) -> list:
        return list(self._chunks)


def is_streamable(value: object) -> bool:
    """True if a handler result should be streamed as chunks."""
    if isinstance(value, (str, bytes, dict, AgentMessage)) or value is None:
        return False
    return isinstance(value, Iterable)


def stream_message(
    msg_type: MessageType,
    correlation_id: str,
    sender: object,
    recipient: object,
    content: object = None,
) -> AgentMessage:
    """Build a STREAM_* message addressed back to the original sender."""
    return AgentMessage(
        type=msg_type,
        sender=sender,
        recipient=recipient,
        content=content,
        correlation_id=correlation_id,
        content_type="application/octet-stream" if content is not None else "text/plain",
    )