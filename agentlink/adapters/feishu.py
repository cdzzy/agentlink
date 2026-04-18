"""
Feishu (Lark) Adapter for AgentLink.

Connects an AgentLink node to the Feishu Bot API so that any agent registered
on the bus can receive and reply to Feishu messages without touching Feishu SDK
internals.

Usage:
    from agentlink.adapters.feishu import FeishuAdapter

    adapter = FeishuAdapter(
        agent_id="my-feishu-bot",
        app_id="cli_xxxx",
        app_secret="xxxxxxxx",
        capabilities=["chat", "qa"],
    )
    bus.register(adapter.as_node())

Webhook endpoint (FastAPI example):
    @app.post("/feishu/event")
    async def feishu_webhook(request: Request):
        return await adapter.handle_webhook(await request.json())

Reference: OpenMantis multi-platform adapter pattern (2026-04-18 AI Trending)
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict, Optional

import httpx

from agentlink.adapters.base import BaseAdapter
from agentlink.protocol.message import AgentMessage


# ─────────────────────────────────────────────────────────────────────────────
# Feishu API constants
# ─────────────────────────────────────────────────────────────────────────────

_FEISHU_BASE = "https://open.feishu.cn/open-apis"
_TOKEN_URL = f"{_FEISHU_BASE}/auth/v3/tenant_access_token/internal"
_REPLY_URL = f"{_FEISHU_BASE}/im/v1/messages/{{message_id}}/reply"
_SEND_URL = f"{_FEISHU_BASE}/im/v1/messages"


class FeishuAdapter(BaseAdapter):
    """
    AgentLink adapter for the Feishu (Lark) Bot platform.

    Translates Feishu webhook events → AgentMessage,
    and AgentMessage replies → Feishu send/reply API calls.
    """

    def __init__(
        self,
        agent_id: str,
        app_id: str,
        app_secret: str,
        namespace: str = "default",
        capabilities: Optional[list] = None,
        description: str = "Feishu bot agent",
        verification_token: Optional[str] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            namespace=namespace,
            capabilities=capabilities or ["chat"],
            description=description,
        )
        self._app_id = app_id
        self._app_secret = app_secret
        self._verification_token = verification_token
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

    # ── Token management ──────────────────────────────────────────────────────

    async def _get_token(self) -> str:
        """Fetch or refresh the tenant_access_token (valid ~2 h)."""
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _TOKEN_URL,
                json={"app_id": self._app_id, "app_secret": self._app_secret},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

        self._access_token = data["tenant_access_token"]
        self._token_expiry = time.time() + data.get("expire", 7200)
        return self._access_token  # type: ignore[return-value]

    # ── Webhook entry point ───────────────────────────────────────────────────

    async def handle_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a Feishu webhook event payload.

        Supports:
        - URL verification challenge
        - im.message.receive_v1 (text messages)

        Returns the appropriate HTTP response body.
        """
        # URL verification handshake
        if "challenge" in payload:
            return {"challenge": payload["challenge"]}

        # Signature verification (optional but recommended)
        if self._verification_token:
            token = payload.get("token", "")
            if not hmac.compare_digest(token, self._verification_token):
                return {"code": 401, "msg": "invalid token"}

        event_type = payload.get("header", {}).get("event_type", "")
        if event_type == "im.message.receive_v1":
            await self._on_message(payload)

        return {"code": 0}

    async def _on_message(self, payload: Dict[str, Any]) -> None:
        """Extract text content and invoke the underlying agent."""
        event = payload.get("event", {})
        msg = event.get("message", {})
        sender = event.get("sender", {}).get("sender_id", {}).get("open_id", "unknown")
        message_id = msg.get("message_id", "")
        chat_id = msg.get("chat_id", "")

        # Only handle plain text for now; extend for cards / files as needed
        import json as _json
        content_raw = msg.get("content", "{}")
        try:
            content_obj = _json.loads(content_raw)
            text = content_obj.get("text", content_raw)
        except Exception:
            text = content_raw

        agent_msg = AgentMessage(
            id=message_id or f"feishu_{int(time.time() * 1000)}",
            role="user",
            content=text,
            metadata={
                "platform": "feishu",
                "sender_open_id": sender,
                "chat_id": chat_id,
                "message_id": message_id,
            },
        )

        reply_text = self._normalize_output(self._invoke(agent_msg))
        if reply_text and message_id:
            await self._reply(message_id, reply_text)

    async def _reply(self, message_id: str, text: str) -> None:
        """Send a reply to a Feishu message thread."""
        token = await self._get_token()
        import json as _json

        async with httpx.AsyncClient() as client:
            await client.post(
                _REPLY_URL.format(message_id=message_id),
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "msg_type": "text",
                    "content": _json.dumps({"text": text}),
                },
                timeout=10,
            )

    # ── BaseAdapter contract ──────────────────────────────────────────────────

    def _invoke(self, message: AgentMessage) -> Any:
        """
        Forward the message to whatever handler is registered on this node.

        In most deployments the node's handler is set externally (via as_node()).
        This default implementation raises — subclass or use as_node() with a
        custom handler instead.
        """
        raise NotImplementedError(
            "FeishuAdapter._invoke: attach a handler via as_node(handler=...) "
            "or subclass and override _invoke."
        )
