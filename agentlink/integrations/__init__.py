"""
AgentLink integrations — bridges to external agent infrastructure.

- engram: long-term memory for the fleet via the MCP stdio protocol
  (agentlink/integrations/engram.py)
"""

from agentlink.integrations.engram import (
    EngramMCPClient,
    EngramMCPError,
    EngramMemoryBackend,
    attach_memory,
)

__all__ = [
    "EngramMCPClient",
    "EngramMCPError",
    "EngramMemoryBackend",
    "attach_memory",
]
