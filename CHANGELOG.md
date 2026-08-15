# Changelog

All notable changes to AgentLink are documented in this file.

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
