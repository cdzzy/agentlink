"""
Message encryption for inter-agent communication (Issue #5).

Encrypt AgentMessage payloads so sensitive data (PII, financial info, trade
secrets) is never transmitted in plaintext. Uses Fernet (AES-128-CBC + HMAC
from the ``cryptography`` package) for authenticated symmetric encryption.

Install: ``pip install cryptography``

Usage::

    from agentlink.security import MessageEncryptor, generate_key

    key = generate_key()
    encryptor = MessageEncryptor(key)

    encrypted = encryptor.encrypt(original_message)
    # wire format: content is the Fernet token, metadata['encrypted'] = True

    decrypted = encryptor.decrypt(encrypted)
    assert decrypted.content == original_message.content
"""

from __future__ import annotations

import json
from typing import Optional

from agentlink.protocol.message import AgentMessage, MessageType


def generate_key() -> str:
    """Generate a new Fernet key (URL-safe base64 string)."""
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode("ascii")


class MessageEncryptor:
    """
    Encrypts and decrypts AgentMessage payloads.

    Args:
        key: A Fernet key (bytes or str), or the name of an env var holding it
             when prefixed with ``env:`` (e.g. ``"env:AGENTLINK_KEY"``).
    """

    def __init__(self, key: Optional[str] = None) -> None:
        from cryptography.fernet import Fernet

        resolved = self._resolve_key(key)
        self._fernet = Fernet(resolved)

    @staticmethod
    def _resolve_key(key: Optional[str]) -> bytes:
        import os

        if key is None:
            key = os.environ.get("AGENTLINK_ENCRYPTION_KEY")
            if key is None:
                raise ValueError(
                    "No encryption key provided. Pass key= or set AGENTLINK_ENCRYPTION_KEY."
                )
        if key.startswith("env:"):
            env_name = key[len("env:"):]
            key = os.environ.get(env_name)
            if key is None:
                raise ValueError(f"Encryption key not found in environment variable {env_name!r}")

        if isinstance(key, bytes):
            return key
        return key.encode("ascii")

    def encrypt(self, message: AgentMessage) -> AgentMessage:
        """
        Encrypt a message, returning an opaque envelope message.

        The returned message's ``content`` is the Fernet token and its
        ``metadata`` carries an ``encrypted`` flag.
        """
        payload = json.dumps(message.to_dict()).encode("utf-8")
        token = self._fernet.encrypt(payload).decode("ascii")
        return AgentMessage(
            type=MessageType.EVENT,
            sender=message.sender,
            recipient=message.recipient,
            content=token,
            content_type="application/agentlink-encrypted",
            metadata={
                **(message.metadata or {}),
                "encrypted": True,
                "encryption": "fernet",
            },
        )

    def decrypt(self, message: AgentMessage) -> AgentMessage:
        """
        Decrypt an envelope produced by :meth:`encrypt`.

        Raises:
            ValueError: If the message is not an encrypted envelope.
        """
        if not message.metadata.get("encrypted"):
            raise ValueError("Message is not encrypted (missing metadata flag)")
        payload = self._fernet.decrypt(message.content.encode("ascii"))
        return AgentMessage.from_dict(json.loads(payload.decode("utf-8")))


def encrypt_message(message: AgentMessage, key: Optional[str] = None) -> AgentMessage:
    """Convenience: encrypt a message with a fresh encryptor."""
    return MessageEncryptor(key).encrypt(message)


def decrypt_message(message: AgentMessage, key: Optional[str] = None) -> AgentMessage:
    """Convenience: decrypt a message with a fresh encryptor."""
    return MessageEncryptor(key).decrypt(message)
