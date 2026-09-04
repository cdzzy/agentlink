"""
Engram memory integration for AgentLink (via MCP).

Gives an AgentBus fleet long-term memory backed by [engram](https://github.com/cdzzy/engram),
communicating over the MCP stdio protocol — no hard coupling: the `engram-mcp`
process can run on Node while AgentLink runs on Python.

Usage::

    from agentlink.integrations.engram import attach_memory

    # Records every routed message into engram and exposes recall()
    backend = attach_memory(bus, command="engram-mcp")
    try:
        bus.send("planner", "worker", "Research AI trends")
        hits = backend.recall_context("AI trends", limit=3)
    finally:
        backend.close()

    # Or drive the client directly
    from agentlink.integrations.engram import EngramMCPClient
    client = EngramMCPClient(command="engram-mcp")
    with client:
        client.store("User prefers dark mode", importance="high")
        hits = client.recall("dark mode")
"""

from __future__ import annotations

import json
import subprocess
import uuid
from typing import Any, Callable, Dict, List, Optional

from agentlink.protocol.message import AgentMessage

MCP_PROTOCOL_VERSION = "2024-11-05"


class EngramMCPError(Exception):
    """Raised when the engram MCP server returns an error or misbehaves."""


class EngramMCPClient:
    """
    Minimal MCP stdio client for an `engram-mcp` server.

    Speaks newline-delimited JSON-RPC 2.0 over the child process's
    stdin/stdout. Use as a context manager::

        with EngramMCPClient(command="engram-mcp") as client:
            client.store("remember this", importance="high")
            hits = client.recall("remember")
    """

    def __init__(
        self,
        command: str = "engram-mcp",
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.command = command
        self.args = args or []
        self.env = env
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> "EngramMCPClient":
        self._process = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=self.env,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._rpc("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "agentlink", "version": "1.0"},
        })
        # MCP handshake notification — no response expected
        self._notify("notifications/initialized", {})
        return self

    def stop(self) -> None:
        if self._process and self._process.stdin:
            try:
                self._process.stdin.close()
            except Exception:
                pass
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None

    def __enter__(self) -> "EngramMCPClient":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ── JSON-RPC plumbing ─────────────────────────────────────────────────

    def _write(self, payload: Dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise EngramMCPError("engram-mcp process is not started")
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()

    def _read(self) -> Optional[Dict[str, Any]]:
        if self._process is None or self._process.stdout is None:
            raise EngramMCPError("engram-mcp process is not started")
        line = self._process.stdout.readline()
        if not line:
            raise EngramMCPError("engram-mcp closed the connection")
        return json.loads(line)

    def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._request_id += 1
        request_id = self._request_id
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            response = self._read()
            if response is None:
                continue
            # Skip notifications/requests that don't answer our id
            if response.get("id") == request_id and ("result" in response or "error" in response):
                break
        if "error" in response:
            raise EngramMCPError(f"MCP error: {response['error']}")
        return response.get("result", {})

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise EngramMCPError(f"Tool '{name}' failed: {result.get('content')}")
        content = result.get("content") or []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return text
        return result

    # ── Engram tool wrappers ──────────────────────────────────────────────

    def store(
        self,
        content: str,
        type: str = "semantic",
        importance: str = "medium",
        tags: Optional[List[str]] = None,
        source: str = "agentlink",
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a memory and return the created memory object."""
        arguments: Dict[str, Any] = {"content": content, "type": type, "importance": importance}
        if tags:
            arguments["tags"] = tags
        if source:
            arguments["source"] = source
        if namespace:
            arguments["namespace"] = namespace
        return self._call_tool("engram_store", arguments)

    def recall(self, keywords: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Multi-signal recall — returns ranked memory dicts."""
        result = self._call_tool("engram_recall", {"keywords": keywords.split(), "limit": limit})
        if isinstance(result, dict):
            return result.get("memories", [])
        return []

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one memory by id."""
        result = self._call_tool("engram_get", {"id": memory_id})
        if isinstance(result, dict) and result.get("id"):
            return result
        return None

    def forget(self, memory_id: str) -> Dict[str, Any]:
        """Permanently delete a memory."""
        return self._call_tool("engram_forget", {"id": memory_id})

    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        return self._call_tool("engram_stats", {})


class EngramMemoryBackend:
    """
    Long-term memory for an AgentBus fleet, backed by engram.

    - ``record_message`` stores routed messages as episodic memories
    - ``recall_context`` retrieves relevant history for prompting
    - ``middleware()`` returns an AgentBus middleware that records automatically
    """

    def __init__(
        self,
        client: EngramMCPClient,
        namespace: str = "agentlink",
        source: str = "agentlink",
        importance: str = "low",
    ) -> None:
        self.client = client
        self.namespace = namespace
        self.source = source
        self.importance = importance  # routed chatter is low-value by default

    def record_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Store a routed message as an episodic memory. Failures are swallowed."""
        summary = message.content if isinstance(message.content, str) else json.dumps(message.content)
        try:
            return self.client.store(
                content=f"{message.type.value} {message.sender}→{message.recipient}: {summary}",
                type="episodic",
                importance=self.importance,
                source=self.source,
                namespace=self.namespace,
            )
        except Exception:
            return None  # memory must never break message routing

    def recall_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Recall relevant memories to enrich an agent's context."""
        return self.client.recall(query, limit=limit)

    def middleware(self) -> Callable[[AgentMessage], Optional[AgentMessage]]:
        """AgentBus middleware that records every routed message."""
        def memory_middleware(message: AgentMessage) -> Optional[AgentMessage]:
            self.record_message(message)
            return message
        return memory_middleware


def attach_memory(
    bus: Any,
    client: Optional[EngramMCPClient] = None,
    command: str = "engram-mcp",
    namespace: str = "agentlink",
) -> EngramMemoryBackend:
    """
    Create an engram memory backend, register its recording middleware on the
    bus, and return the backend. Call ``backend.client.stop()`` (or use the
    client as a context manager) on shutdown.
    """
    if client is None:
        client = EngramMCPClient(command=command).start()
    backend = EngramMemoryBackend(client, namespace=namespace)
    bus.use(backend.middleware())
    return backend
