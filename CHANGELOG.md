# Changelog

All notable changes to AgentLink are documented in this file.

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
