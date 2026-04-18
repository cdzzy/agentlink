"""
WeCom (企业微信) Adapter for AgentLink.

Connects an AgentLink node to the WeCom Bot / Application Message API,
handling XML event parsing, passive reply, and active push.

Usage:
    from agentlink.adapters.wecom import WeComAdapter

    adapter = WeComAdapter(
        agent_id="my-wecom-bot",
        corp_id="ww_xxxx",
        corp_secret="xxxxxxxx",
        agent_id_num=1000002,   # WeCom application agent_id (integer)
        token="your_token",
        encoding_aes_key="your_aes_key",
        capabilities=["chat", "notify"],
    )
    bus.register(adapter.as_node())

Webhook (FastAPI):
    @app.get("/wecom/event")
    def verify(msg_signature, timestamp, nonce, echostr):
        return adapter.verify(msg_signature, timestamp, nonce, echostr)

    @app.post("/wecom/event")
    async def event(request: Request, msg_signature, timestamp, nonce):
        return await adapter.handle_webhook(
            await request.body(), msg_signature, timestamp, nonce
        )

Reference: OpenMantis multi-platform adapter (2026-04-18 AI Trending)
"""

from __future__ import annotations

import hashlib
import hmac
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional

import httpx

from agentlink.adapters.base import BaseAdapter
from agentlink.protocol.message import AgentMessage


_WECOM_BASE = "https://qyapi.weixin.qq.com/cgi-bin"
_TOKEN_URL = f"{_WECOM_BASE}/gettoken"
_SEND_URL = f"{_WECOM_BASE}/message/send"


class WeComAdapter(BaseAdapter):
    """
    AgentLink adapter for WeCom (企业微信) Application Messages.

    Handles:
    - GET verification (echostr)
    - POST encrypted XML event → AgentMessage → reply
    - Active push via send_message()
    """

    def __init__(
        self,
        agent_id: str,
        corp_id: str,
        corp_secret: str,
        agent_id_num: int,
        token: str,
        encoding_aes_key: str,
        namespace: str = "default",
        capabilities: Optional[list] = None,
        description: str = "WeCom agent",
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            namespace=namespace,
            capabilities=capabilities or ["chat"],
            description=description,
        )
        self._corp_id = corp_id
        self._corp_secret = corp_secret
        self._agent_id_num = agent_id_num
        self._token = token
        self._encoding_aes_key = encoding_aes_key

        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

    # ── Token ─────────────────────────────────────────────────────────────────

    async def _get_token(self) -> str:
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                _TOKEN_URL,
                params={"corpid": self._corp_id, "corpsecret": self._corp_secret},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        self._access_token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 7200)
        return self._access_token  # type: ignore[return-value]

    # ── Signature verification ────────────────────────────────────────────────

    def _check_signature(self, msg_signature: str, timestamp: str, nonce: str, encrypt: str = "") -> bool:
        """Verify WeCom message signature."""
        arr = sorted([self._token, timestamp, nonce, encrypt])
        sha = hashlib.sha1("".join(arr).encode()).hexdigest()
        return hmac.compare_digest(sha, msg_signature)

    def verify(self, msg_signature: str, timestamp: str, nonce: str, echostr: str) -> str:
        """Handle GET verification request — return echostr if signature valid."""
        if self._check_signature(msg_signature, timestamp, nonce):
            # In production, decrypt echostr with AES key; simplified here
            return echostr
        raise ValueError("WeCom signature verification failed")

    # ── Webhook ───────────────────────────────────────────────────────────────

    async def handle_webhook(
        self,
        body: bytes,
        msg_signature: str,
        timestamp: str,
        nonce: str,
    ) -> str:
        """
        Parse encrypted WeCom XML, invoke agent, return passive XML reply.

        Returns an empty string if no reply is needed (e.g., event-only messages).
        """
        try:
            root = ET.fromstring(body.decode("utf-8"))
        except ET.ParseError:
            return ""

        # For simplicity, handle un-encrypted XML (set to 0 encoding mode)
        # Production: decrypt <Encrypt> field using AES key
        msg_type = root.findtext("MsgType", "")
        from_user = root.findtext("FromUserName", "unknown")
        agent_id_in = root.findtext("AgentID", "")

        if msg_type == "text":
            content = root.findtext("Content", "")
            msg_id = root.findtext("MsgId", f"wecom_{int(time.time()*1000)}")

            agent_msg = AgentMessage(
                id=msg_id,
                role="user",
                content=content,
                metadata={
                    "platform": "wecom",
                    "from_user": from_user,
                    "agent_id": agent_id_in,
                },
            )

            reply_text = self._normalize_output(self._invoke(agent_msg))

            if reply_text:
                now = int(time.time())
                return (
                    f"<xml>"
                    f"<ToUserName><![CDATA[{from_user}]]></ToUserName>"
                    f"<FromUserName><![CDATA[{self.agent_id}]]></FromUserName>"
                    f"<CreateTime>{now}</CreateTime>"
                    f"<MsgType><![CDATA[text]]></MsgType>"
                    f"<Content><![CDATA[{reply_text}]]></Content>"
                    f"</xml>"
                )

        return ""

    # ── Active push ───────────────────────────────────────────────────────────

    async def send_message(self, to_user: str, text: str) -> Dict[str, Any]:
        """Proactively push a text message to a WeCom user."""
        token = await self._get_token()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _SEND_URL,
                params={"access_token": token},
                json={
                    "touser": to_user,
                    "msgtype": "text",
                    "agentid": self._agent_id_num,
                    "text": {"content": text},
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()

    # ── BaseAdapter ───────────────────────────────────────────────────────────

    def _invoke(self, message: AgentMessage) -> Any:
        raise NotImplementedError(
            "WeComAdapter._invoke: attach via as_node(handler=...) or subclass."
        )





