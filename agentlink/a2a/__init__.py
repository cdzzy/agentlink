"""
AgentLink A2A v1.0 compatibility layer.

Bridges AgentLink agents to the A2A (Agent2Agent) protocol v1.0:

  * :mod:`agentlink.a2a.card`     — Agent Card generation/parsing for the
    ``/.well-known/agent-card.json`` discovery document.
  * :mod:`agentlink.a2a.mapping`  — lossless AgentMessage <-> A2A
    Message/Task/Artifact mapping (the AgentLink envelope rides in A2A
    ``metadata["agentlink.envelope"]``).
  * :mod:`agentlink.a2a.transport` — JSON-RPC 2.0 client/server bridging
    (:class:`A2ATransport`).
  * :mod:`agentlink.a2a.verify`   — inbound Agent Card JWKS signature
    verification (requires the optional ``a2a`` extras; import guarded).

Only ``agentlink.a2a.verify`` needs third-party packages (``pyjwt`` +
``cryptography``, exposed as the optional ``[a2a]`` extra); everything else is
standard library only.
"""

from agentlink.a2a.card import (
    A2A_PROTOCOL_VERSION,
    AGENT_CARD_WELLKNOWN_PATH,
    DEFAULT_CARD_VERSION,
    AgentCapabilities,
    AgentCard,
    AgentCardSignature,
    AgentInterface,
    AgentSkill,
    agent_card_url,
)
from agentlink.a2a.mapping import (
    ENVELOPE_METADATA_KEY,
    TASK_STATE_AUTH_REQUIRED,
    TASK_STATE_CANCELED,
    TASK_STATE_COMPLETED,
    TASK_STATE_FAILED,
    TASK_STATE_INPUT_REQUIRED,
    TASK_STATE_REJECTED,
    TASK_STATE_SUBMITTED,
    TASK_STATE_WORKING,
    a2a_message_to_agent_message,
    a2a_task_to_agent_message,
    agent_message_to_a2a_message,
    agent_message_to_a2a_task,
    content_to_part,
    has_agentlink_envelope,
    is_terminal_state,
    parts_to_content,
    strip_envelope,
    task_reply_to_a2a_task,
)
from agentlink.a2a.transport import A2ATransport, A2ATransportError

# Card verification needs pyjwt + cryptography (optional extras).
try:
    from agentlink.a2a.verify import (  # noqa: F401
        A2AVerificationError,
        canonical_card_bytes,
        generate_signing_key,
        private_key_to_jwks,
        sign_agent_card,
        verify_agent_card,
    )
    _a2a_verify_available = True
except ImportError:
    _a2a_verify_available = False

__all__ = [
    # card
    "AGENT_CARD_WELLKNOWN_PATH",
    "A2A_PROTOCOL_VERSION",
    "DEFAULT_CARD_VERSION",
    "AgentCard",
    "AgentCardSignature",
    "AgentInterface",
    "AgentSkill",
    "AgentCapabilities",
    "agent_card_url",
    # mapping
    "ENVELOPE_METADATA_KEY",
    "TASK_STATE_AUTH_REQUIRED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_REJECTED",
    "TASK_STATE_SUBMITTED",
    "TASK_STATE_WORKING",
    "agent_message_to_a2a_message",
    "a2a_message_to_agent_message",
    "agent_message_to_a2a_task",
    "a2a_task_to_agent_message",
    "task_reply_to_a2a_task",
    "content_to_part",
    "parts_to_content",
    "has_agentlink_envelope",
    "strip_envelope",
    "is_terminal_state",
    # transport
    "A2ATransport",
    "A2ATransportError",
]

if _a2a_verify_available:
    __all__.extend([
        "A2AVerificationError",
        "canonical_card_bytes",
        "generate_signing_key",
        "private_key_to_jwks",
        "sign_agent_card",
        "verify_agent_card",
    ])
