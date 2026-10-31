# Changelog

All notable changes to AgentLink are documented in this file.

## [0.9.0] - 2026-11-01

### Added

- **MCP Streamable HTTP transport** (`create_streamable_mcp_app`): the MCP
  server side now implements the transport modern MCP clients (Claude
  Desktop, Cursor, MCP Inspector) expect, per the 2025-03-26 spec revision
  and refined through 2025-11-25. `create_mcp_app` (plain JSON-RPC over POST)
  is unchanged for minimal setups.
  - **Content negotiation** — `POST` replies as `application/json` by
    default, or as an `event: message` SSE frame when the client's `Accept`
    only allows `text/event-stream`.
  - **Session management** — `initialize` mints an `Mcp-Session-Id` returned
    in the response header and required on follow-up requests. Unknown ids
    yield `404`, `DELETE` terminates a session (`204`), and
    `session_required=True` rejects requests that omit the header. Sessions
    are per-app, so one process can serve several clients.
  - **SSE channel** — `GET` with `Accept: text/event-stream` opens the
    server-to-client stream for a known session; a plain `GET` returns server
    metadata.
  - **CORS** — `OPTIONS` preflight answered with the MCP headers exposed, so
    browser-based clients can connect.
  - `session_required` and a custom endpoint `path` are configurable;
    `MCP_SESSION_HEADER` / `MCP_PROTOCOL_HEADER` constants are exported.

### Fixed

- **`tools/list` (and siblings) violated the MCP spec shape** — the server
  returned a bare array while `MCPAdapter.list_tools()` on the client side
  already expected `{"tools": [...]}`, so server and client disagreed.
  `tools/list`, `resources/list`, `prompts/list` and `resources/read` now
  return the spec-required `{"tools" | "resources" | "prompts" | "contents":
  [...]}` wrappers. A spec-compliant MCP client can now parse the responses.

### Changed

- **New optional extra** `mcp` (`pip install cdzzy-agentlink[mcp]`) declaring
  `aiohttp>=3.9`, which `create_mcp_app` / `create_streamable_mcp_app` need.
  It was previously an undeclared implicit dependency. Also added to `all`.
- `agentlink/__init__.py` now exports the server surface:
  `MCPServerAdapter`, `create_mcp_app`, `create_streamable_mcp_app`,
  `expose_bus_as_mcp`, `MCP_SESSION_HEADER`, `MCP_PROTOCOL_HEADER`.
- README (both languages) gained an "Exposing Agents as an MCP Server"
  section; the Chinese roadmap item for MCP compatibility is marked done.

### Testing

- 18 new tests (`tests/test_mcp_streamable_http.py`): session lifecycle,
  content negotiation, SSE channel, CORS, and JSON-RPC error paths.
- `tests/test_fastmcp_adapter.py` updated for the spec-shaped `tools/list`.
- Full suite: **277 passed, 14 skipped**; `ruff check .` and `mypy agentlink`
  clean.

## [0.8.3] - 2026-10-23

### Fixed

- **A2A discovery never awaited the agent card fetch**
  (`agentlink/adapters/a2a_adapter.py`): `A2AAdapter.discover_and_wrap()` called
  the async `get_agent_card()` without `await`, so `card.name` crashed at
  runtime. The method is now `async def` and awaits the card fetch.
- **A2A `agent/sendMessage` crashed before reaching the node**
  (`agentlink/adapters/a2a_adapter.py`): the forwarded `AgentMessage` omitted
  the required `sender`/`recipient` fields (raising `TypeError`), and the node
  was invoked via a non-existent `send_sync` method (`AttributeError`). The
  server now builds a properly-addressed message, routes it through the node's
  message handler, and reports "no reply" instead of crashing.
- **`HubRegistry.all_live()` advertised the wrong return type**
  (`agentlink/hub.py`): it declared `List[Registration]` but returned serialized
  dicts; callers (`discover`, the hub HTTP endpoint) index them as dicts. The
  annotation now matches the actual `List[Dict[str, Any]]`.
- **WebSocket transport type errors** (`agentlink/transport.py`): the
  connection/server slots were inferred as `None`, and error replies passed
  bare strings where `AgentAddress` is required; both are now explicitly typed /
  parsed addresses.

### Changed

- **mypy now runs in CI** (`.github/workflows/ci.yml`): the type checker was
  configured in `pyproject.toml` but never executed. The lint job now installs
  mypy and checks the `agentlink` package. `[tool.mypy]` moves to
  `python_version = "3.10"` (the mypy 2.x floor; 3.9 runtime compat is kept via
  `from __future__ import annotations`).
- **Type-correctness fixes across adapters**: `LangGraphAdapter`/`CrewAIAdapter`/
  `AutoGenAdapter` now declare `capabilities` as
  `Optional[List[Union[str, AgentCapability]]]` (matching `BaseAdapter`), the
  JWK converter uses a branch-local `ec_numbers` name instead of reusing the RSA
  variable, and RSA/EdDSA key generation carries a scoped
  `# type: ignore[assignment]` for the cryptography key-union.
- **CI concurrency guard & least privilege** (`.github/workflows/ci.yml`):
  push/PR runs now cancel stale in-progress runs for the same ref and the
  workflow only requests `contents: read`.

## [0.8.2] - 2026-10-16

### Fixed

- **`__version__` could be poisoned by stale local `*.egg-info` dirs**
  (`agentlink/__init__.py`): the previous metadata-first resolution returned
  an old version (e.g. `0.6.0`) on machines carrying leftover develop-install
  `*.egg-info` folders, even though the checkout was at `0.8.1`. Version
  resolution now prefers the adjacent `pyproject.toml` (single source of
  truth) and only falls back to `importlib.metadata` for regular installed
  copies, so `__version__` can never drift or go stale.

### Changed

- **Python 3.13 classifier**: the declared supported-Python list in
  `pyproject.toml` now includes 3.13, matching the already-green CI matrix
  (tests were passing on 3.13 but the classifier advertised only up to 3.12).
- **Modernized type annotations** in `agentlink/transport.py` and
  `agentlink/adapters/base.py`: `typing.Dict` / `typing.Optional` /
  `typing.List` / `typing.Union` were replaced with PEP 585 built-in generics
  and PEP 604 union syntax (`dict[str, Any]`, `str | None`,
  `list[str | AgentCapability]`), which are valid on Python 3.9 via
  `from __future__ import annotations`.

### Added tests

- `tests/test_version.py`: guards that `agentlink.__version__` always equals
  the `pyproject.toml` version and stays PEP 440-valid.

## [0.8.1] - 2026-10-02

### Fixed

- **WebSocket transport robustness** (`agentlink.transport`): the server handler now tolerates malformed JSON frames (replies with an ERROR frame instead of crashing the connection), awaits async `on_message` callbacks (the `WSBridge` async handler previously crashed the handler thread on first reply), and the client `send()` no longer blocks forever against a silent server — replies time out after `reply_timeout` seconds (default 30s).
- **Hub hardening** (`agentlink.hub`): POST bodies are capped at 1 MiB (413 on oversized, 400 on invalid JSON/Content-Length instead of a handler crash), and the heartbeat loop re-registers the client after 3 consecutive failures so agents survive a hub restart instead of silently vanishing from discovery.
- **Version drift**: `agentlink.__version__` is now sourced from installed package metadata (with a fallback), so it can no longer lag behind `pyproject.toml`.

### Changed

- CI: `actions/setup-python` now caches pip installs across the lint and 5-version test matrix.
- Added `tests/test_transport.py` covering request/reply round-trips, the reply timeout, and malformed-frame tolerance (4 new tests).

## [0.8.0] - 2026-10-02

### Added

- **A2A v1.0 SSE streaming** (`agentlink.a2a`): full `SendStreamingMessage` support on both sides of the wire.
  - *Client*: `A2ATransport.stream()` / `astream()` POST with `Accept: text/event-stream`, parse the SSE body event-by-event and return reply `AgentMessage`s in arrival order; `build_stream_request()` emits the v1.0 method (legacy `message/stream` still accepted).
  - *Server*: `parse_stream_request()` decodes streaming requests; `build_stream_events()` / `build_sse_body()` / `build_stream_response()` emit the v1.0 *task-lifecycle stream* pattern — opening `Task` (`kind: "task"`) → a pure-state `status-update` (`WORKING`) before every chunk → one `artifact-update` per chunk (final chunk carries `lastChunk: true`) → terminal `status-update` (`final: true`). Content travels only via artifacts, so nothing is decoded twice.
  - *Mapping*: `task_object`, `status_update_event`, `artifact_update_event`, `stream_event_to_agent_message` plus the `EVENT_KIND_*` discriminators. Status updates can embed an envelope-carrying Message for lossless AgentLink-to-AgentLink streaming; the opening Task and unknown event kinds decode to `None`.
- 32 streaming tests covering event construction/decoding, SSE parsing (keep-alives, multi-line `data:` blocks, JSON-RPC error bodies), event-pattern shape and full server→client wire round-trips.

### Changed

- **v1.0 conformance of wire objects**: every serialized `Message` now carries `kind: "message"`, and every `Part` is a proper discriminated union — `{"kind": "text", ...}` / `{"kind": "data", ...}` / `{"kind": "file", "file": {"fileWithBytes": ..., "mimeType": ...}}`. `parts_to_content` still tolerates the legacy shapes, so old peers keep interoperating.

## [0.7.1] - 2026-09-25

### Fixed

- `examples/05_async_streaming.py` referenced an undefined `AgentAddress` symbol (ruff F821) and built a dead `AgentMessage` that was never sent; the demo now streams directly.
- Examples linted repo-wide: 34 ruff findings fixed across `examples/` — unused imports/variables, unsorted imports, placeholder-less f-strings, and dead bindings. The A2A server example now prints its agent card JSON instead of leaving `server` unused.

### Changed

- CI lint job pins `ruff==0.16.8` for reproducible results and lints the whole repository instead of only `agentlink tests`, so future example regressions (like the F821 above) are caught on the first commit.

## [0.7.0] - 2026-09-20

### Added

- **A2A v1.0 compatibility layer** (`agentlink.a2a`): bridge AgentLink agents to the Agent2Agent protocol v1.0 without leaving the bus — `AgentCard` generation/parsing for the `/.well-known/agent-card.json` discovery document, lossless `AgentMessage ⇄ A2A Message/Task/Artifact` mapping (the original envelope rides in `metadata["agentlink.envelope"]`), and an `A2ATransport` JSON-RPC 2.0 client. Card signing/verification (`sign_agent_card` / `verify_agent_card`, detached JWS + JWKS) lives behind the optional `pip install agentlink[a2a]` extra; everything else is stdlib-only.
- **A2A HTTP adapter** (`adapters/a2a_adapter.py`): expose a running node over A2A, or call remote A2A agents from the bus.
- **Release automation** (`.github/workflows/release.yml`): PyPI publish on `v*` tags, guarded on `PYPI_API_TOKEN`.

### Changed

- CI: dedicated ruff lint job (132 findings fixed across adapters/runtime/tests) and a 3.9–3.13 test matrix with fail-fast off.

### Fixed

- JWKS key selection during card verification now probes every published key when the JWS header omits `kid`, instead of failing with "No JWKS key matches kid None".

## [0.6.0] - 2026-09-04

### Added

- **AgentLink Hub — distributed registry** (`agentlink/hub.py`): zero-dependency HTTP registry where agents across processes/networks announce themselves and discover peers by capability. Heartbeat-based liveness (TTL expiry), token-authenticated registration, `HubServer` + `HubClient` (background heartbeat thread, context-manager lifecycle).

## [0.5.0] - 2026-09-04

### Added

- **Long-term memory via engram** (`agentlink/integrations/engram.py`): `attach_memory(bus)` records every routed message into [engram](https://github.com/cdzzy/engram) through its MCP stdio server (`engram-mcp`) — language-agnostic bridge with no hard coupling. `EngramMCPClient` exposes `store` / `recall` / `get` / `forget` / `stats`; `EngramMemoryBackend.recall_context()` retrieves shared context for prompting. Memory failures never break message routing.

## [0.4.0] - 2026-08-27

### Added

- **OpenTelemetry-compatible tracing**: `instrument_bus(bus, service_name=, exporter=|tracer=)` wraps message routing with spans covering delivery and replies. Zero-dependency `InMemorySpanExporter` for tests/dashboards, or pass any `opentelemetry.trace.Tracer` to export through your OTel pipeline. Error spans recorded on delivery failures.

## [0.3.0] - 2026-08-19

### Added

- **Streaming for long-running tasks**: handlers that return an iterable are automatically streamed back as `STREAM_START` / `STREAM_CHUNK` / `STREAM_END` messages. `AgentNode.stream()` returns a `StreamResult` that iterates chunks as they arrive and exposes `.collect()` for the joined text. Streams are correlated by request id so concurrent streams never interleave.

## [0.2.0] - 2026-08-15

### Added

- **Structured message schemas** (`#1`): `MessageSchema` + `SchemaRegistry` for opt-in runtime validation, integrated via `AgentBus.register_schema` / `validate_message` and `AgentNode.send(..., schema=...)`.
- **Dead letter queue** (`#2`): `DeadLetterQueue` + `AgentBus(dlq_enabled=, max_retries=, dlq_handler=)`. Error replies are retried then dead-lettered; `bus.dlq.retry()` for manual recovery.
- **Message encryption** (`#5`): `MessageEncryptor` (Fernet/AES) with `generate_key`, `encrypt_message`, `decrypt_message`. Key from argument or `AGENTLINK_ENCRYPTION_KEY` env var.
- **WebSocket transport** (`#4`): `WSTransport` (server/client) + `WSBridge` to connect a transport to an `AgentBus`.
- **Protocol gateway** (`#6`): `ProtocolGateway` for routing across heterogeneous protocols with per-protocol adapters and `@gateway.on()` receive handlers.

### Changed

- A2A protocol adapter (`#3`) confirmed already present (`agentlink/adapters/a2a_adapter.py`); no changes needed.
- Fixed a pre-existing test bug (missing `create_mcp_app` import) and an outdated aiohttp test API usage.

## [0.1.0]

- Initial release: AgentMessage protocol, AgentBus, AgentNode, framework adapters, MCP adapter, A2A adapter.
