"""
AgentMessage ⇄ A2A v1.0 Task/Artifact mapping (AgentLink A2A compatibility layer).

Pure, dependency-free conversion between the AgentLink protocol envelope
(:class:`agentlink.protocol.message.AgentMessage`) and the A2A v1.0 data
model (``Task``, ``Message``, ``Artifact``, unified ``Part``).

Lossless round-trip strategy
----------------------------
An AgentMessage has fields with no direct A2A equivalent (sender/recipient
addresses, ``type``, ``parent_id``, ``ttl``, ``content_type``, ...). To keep
``send``/``receive`` round-trips lossless, every outbound A2A message carries
the full AgentLink envelope inside its ``metadata`` under the reserved key
``agentlink.envelope`` (``ENVELOPE_METADATA_KEY``). Native A2A peers simply
ignore it; AgentLink peers use it to rebuild the message byte-for-byte.

Content mapping (A2A v1.0 unified ``Part`` — one of ``text``/``data``/``raw``/``url``):
  - ``str``                        → ``{"text": ..., "mediaType": "text/plain"}``
  - ``dict`` / ``list`` / scalars  → ``{"data": ..., "mediaType": "application/json"}``
  - ``bytes`` / ``bytearray``      → ``{"raw": base64, "mediaType": "application/octet-stream"}``
  - ``None``                       → ``{"data": None}``

v1.0 conventions: roles are ``ROLE_USER``/``ROLE_AGENT``, task states are
``TASK_STATE_*`` (SCREAMING_SNAKE_CASE). Parsing is tolerant of the legacy
v0.2/v0.3 wire format (``"user"``/``"agent"`` roles, ``kind``-discriminated
parts, lowercase task states).
"""

from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agentlink.protocol.message import AgentAddress, AgentMessage, MessageType

A2A_PROTOCOL_VERSION = "1.0"

# Reserved metadata key holding the AgentLink envelope on the A2A wire.
ENVELOPE_METADATA_KEY = "agentlink.envelope"
ENVELOPE_VERSION = 1

# Task states (v1.0 SCREAMING_SNAKE_CASE with prefix).
TASK_STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
TASK_STATE_WORKING = "TASK_STATE_WORKING"
TASK_STATE_COMPLETED = "TASK_STATE_COMPLETED"
TASK_STATE_FAILED = "TASK_STATE_FAILED"
TASK_STATE_CANCELED = "TASK_STATE_CANCELED"
TASK_STATE_REJECTED = "TASK_STATE_REJECTED"
TASK_STATE_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
TASK_STATE_AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"

_TERMINAL_STATES = {
    TASK_STATE_COMPLETED, TASK_STATE_FAILED, TASK_STATE_CANCELED, TASK_STATE_REJECTED,
}


def now_iso_ms() -> str:
    """Current UTC time as ISO 8601 with millisecond precision (A2A v1.0 format)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def is_terminal_state(state: str) -> bool:
    """Return True if the (v1.0 or legacy) task state is terminal."""
    return _normalize_state(state) in _TERMINAL_STATES


def _normalize_state(state: str) -> str:
    """Normalize a task state to the v1.0 SCREAMING_SNAKE_CASE form."""
    s = (state or "").strip().upper().replace("-", "_")
    if not s.startswith("TASK_STATE_"):
        s = "TASK_STATE_" + s
    return s


# ---------------------------------------------------------------------------
# Content ⇄ Part
# ---------------------------------------------------------------------------

def content_to_part(content: Any) -> Dict[str, Any]:
    """Convert an AgentMessage ``content`` into one A2A v1.0 ``Part``."""
    if isinstance(content, str):
        return {"text": content, "mediaType": "text/plain"}
    if isinstance(content, (dict, list, bool, int, float)) or content is None:
        return {"data": content, "mediaType": "application/json"}
    if isinstance(content, (bytes, bytearray)):
        return {
            "raw": base64.b64encode(bytes(content)).decode("ascii"),
            "mediaType": "application/octet-stream",
        }
    raise ValueError(
        f"A2A mapping supports JSON-serializable content (str/dict/list/scalars/bytes); "
        f"got {type(content).__name__}. Serialize it first."
    )


def parts_to_content(parts: List[Dict[str, Any]]) -> Any:
    """
    Rebuild an AgentMessage ``content`` from A2A ``Part`` objects.

    Uses the first part; consecutive text parts are joined with newlines.
    Tolerates legacy v0.2/v0.3 ``kind``-discriminated parts.
    """
    texts: List[str] = []
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if "text" in part:                                   # v1.0 text part
            texts.append(part["text"])
        elif "data" in part:                                 # v1.0 data part
            return part["data"]
        elif "raw" in part:                                  # v1.0 raw bytes part
            return base64.b64decode(part["raw"])
        elif "url" in part:                                  # v1.0 file reference part
            return {"url": part["url"], "mediaType": part.get("mediaType")}
        elif part.get("kind") == "file" and "file" in part:  # legacy v0.3 file part
            f = part["file"] or {}
            uri = f.get("fileWithUri", f.get("fileWithBytes"))
            return {"url": uri, "mediaType": f.get("mimeType")}
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Envelope embedding
# ---------------------------------------------------------------------------

def _envelope_dict(message: AgentMessage) -> Dict[str, Any]:
    return {"v": ENVELOPE_VERSION, "message": message.to_dict()}


def has_agentlink_envelope(metadata: Optional[Dict[str, Any]]) -> bool:
    """Return True if the A2A metadata carries an AgentLink envelope."""
    env = (metadata or {}).get(ENVELOPE_METADATA_KEY)
    return isinstance(env, dict) and isinstance(env.get("message"), dict)


def _from_envelope(metadata: Optional[Dict[str, Any]]) -> Optional[AgentMessage]:
    """Rebuild an AgentMessage from its embedded envelope, if present."""
    env = (metadata or {}).get(ENVELOPE_METADATA_KEY)
    if isinstance(env, dict) and env.get("v") == ENVELOPE_VERSION \
            and isinstance(env.get("message"), dict):
        try:
            return AgentMessage.from_dict(env["message"])
        except (KeyError, ValueError, TypeError):
            return None
    return None


def strip_envelope(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the A2A metadata without the reserved envelope key."""
    return {k: v for k, v in (metadata or {}).items() if k != ENVELOPE_METADATA_KEY}


# ---------------------------------------------------------------------------
# AgentMessage ⇄ A2A Message
# ---------------------------------------------------------------------------

def agent_message_to_a2a_message(message: AgentMessage) -> Dict[str, Any]:
    """Serialize an AgentMessage into an A2A v1.0 ``Message`` object."""
    metadata = dict(message.metadata or {})
    metadata[ENVELOPE_METADATA_KEY] = _envelope_dict(message)

    out: Dict[str, Any] = {
        "messageId": message.id,
        "role": "ROLE_AGENT" if message.type is MessageType.REPLY else "ROLE_USER",
        "parts": [content_to_part(message.content)],
        "metadata": metadata,
        # Stable task id derived from the AgentLink message id.
        "taskId": message.id,
    }
    if message.correlation_id:
        out["contextId"] = message.correlation_id
    return out


def a2a_message_to_agent_message(
    a2a_message: Dict[str, Any],
    *,
    default_sender: Optional[AgentAddress] = None,
    default_recipient: Optional[AgentAddress] = None,
) -> AgentMessage:
    """
    Rebuild an AgentMessage from an A2A ``Message`` object.

    If the embedded AgentLink envelope is present, the message is restored
    losslessly; otherwise the message is synthesized from the A2A fields
    (role, parts, ids, metadata).
    """
    restored = _from_envelope(a2a_message.get("metadata"))
    if restored is not None:
        return restored

    metadata = strip_envelope(a2a_message.get("metadata"))
    role = str(a2a_message.get("role", "user")).lower().removeprefix("role_")
    msg_type = MessageType.REPLY if role == "agent" else MessageType.REQUEST
    content = parts_to_content(a2a_message.get("parts") or [])

    return AgentMessage(
        type=msg_type,
        sender=default_sender or AgentAddress("a2a-client", "a2a"),
        recipient=default_recipient or AgentAddress("a2a-agent", "a2a"),
        content=content,
        id=a2a_message.get("messageId") or str(uuid.uuid4()),
        correlation_id=a2a_message.get("contextId"),
        metadata=metadata,
        content_type="text/plain" if isinstance(content, str) else "application/json",
    )


# ---------------------------------------------------------------------------
# AgentMessage ⇄ A2A Task (with artifacts)
# ---------------------------------------------------------------------------

def agent_message_to_a2a_task(
    message: AgentMessage,
    state: str = TASK_STATE_SUBMITTED,
) -> Dict[str, Any]:
    """
    Serialize an AgentMessage into an A2A v1.0 ``Task`` (status ``submitted``).

    The full envelope travels inside ``history[0].metadata``; the AgentLink
    message id is reused as the task id for end-to-end correlation.
    """
    return {
        "id": message.id,
        "contextId": message.correlation_id or message.id,
        "status": {"state": state, "timestamp": now_iso_ms()},
        "history": [agent_message_to_a2a_message(message)],
        "metadata": strip_envelope(message.metadata),
    }


def a2a_task_to_agent_message(
    task: Dict[str, Any],
    original: Optional[AgentMessage] = None,
) -> AgentMessage:
    """
    Rebuild an AgentMessage (a reply) from an A2A ``Task``.

    Resolution order:
      1. AgentLink envelope in ``history[-1].metadata`` → lossless restore.
      2. AgentLink envelope in ``task.metadata`` (fallback placement).
      3. Artifacts / history text + ``original.reply(...)`` correlation.
    """
    # 1) envelope from the last history message (where we embed the reply)
    history = task.get("history") or []
    for entry in reversed(history):
        metadata = (entry or {}).get("metadata") if isinstance(entry, dict) else None
        restored = _from_envelope(metadata)
        if restored is not None:
            return restored

    # 2) envelope stashed at task level
    restored = _from_envelope(task.get("metadata"))
    if restored is not None:
        return restored

    # 3) synthesize a reply from artifacts / history
    content = _content_from_artifacts(task)
    if content is None and history:
        last = history[-1] if isinstance(history[-1], dict) else {}
        content = parts_to_content(last.get("parts") or [])
    if original is not None:
        reply = original.reply(content if content is not None else "")
        reply.id = task.get("id") or reply.id
        reply.metadata = strip_envelope(task.get("metadata"))
        return reply

    sender = AgentAddress(str(task.get("id", "a2a-agent")), "a2a")
    return AgentMessage(
        type=MessageType.REPLY,
        sender=sender,
        recipient=AgentAddress("a2a-client", "a2a"),
        content=content if content is not None else "",
        id=task.get("id") or str(uuid.uuid4()),
        correlation_id=task.get("contextId"),
        metadata=strip_envelope(task.get("metadata")),
        content_type="text/plain" if isinstance(content, str) else "application/json",
    )


def task_reply_to_a2a_task(
    original_task: Dict[str, Any],
    reply: AgentMessage,
    *,
    artifact_name: str = "result",
) -> Dict[str, Any]:
    """
    Build the completed A2A ``Task`` answering ``original_task``.

    The reply content is exposed as a native ``Artifact`` (for foreign A2A
    clients) while the reply envelope rides in ``history[0].metadata`` (for
    lossless AgentLink round-trips).
    """
    task = agent_message_to_a2a_task(reply, state=TASK_STATE_COMPLETED)
    task["id"] = original_task.get("id") or task["id"]
    task["contextId"] = original_task.get("contextId") or task["contextId"]
    task["artifacts"] = [{
        "artifactId": str(uuid.uuid4()),
        "name": artifact_name,
        "parts": [content_to_part(reply.content)],
        "metadata": {},
    }]
    return task


def _content_from_artifacts(task: Dict[str, Any]) -> Any:
    for artifact in task.get("artifacts") or []:
        if isinstance(artifact, dict) and artifact.get("parts"):
            return parts_to_content(artifact["parts"])
    return None
