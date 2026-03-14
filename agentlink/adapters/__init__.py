"""
AgentLink Adapters — bridge between AgentLink protocol and agent frameworks.

Each adapter wraps a framework-specific agent into an AgentLink AgentNode,
handling the translation between AgentLink messages and the framework's
native input/output format.
"""

from agentlink.adapters.base import BaseAdapter
from agentlink.adapters.langgraph_adapter import LangGraphAdapter
from agentlink.adapters.autogen_adapter import AutoGenAdapter
from agentlink.adapters.crewai_adapter import CrewAIAdapter
from agentlink.adapters.generic import GenericAdapter

__all__ = [
    "BaseAdapter",
    "LangGraphAdapter",
    "AutoGenAdapter",
    "CrewAIAdapter",
    "GenericAdapter",
]
