"""
AgentLink Capability System.

Agents declare what they can do using capabilities.
The bus uses capabilities for semantic routing:
"send this to any agent that can do web-search"
rather than hard-coded agent IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentCapability:
    """
    Declares a single capability an agent has.

    Example:
        cap = AgentCapability(
            name="web-search",
            description="Search the web for current information",
            input_schema={"type": "string", "description": "search query"},
            output_schema={"type": "string", "description": "search results"},
            version="1.0",
        )
    """
    name: str
    description: str = ""
    version: str = "1.0"
    input_schema: Optional[Dict[str, Any]] = None    # JSON Schema
    output_schema: Optional[Dict[str, Any]] = None   # JSON Schema
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, query: str) -> bool:
        """Check if this capability matches a query string (name or tag)."""
        q = query.lower()
        if q == self.name.lower():
            return True
        if any(q == t.lower() for t in self.tags):
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCapability":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CapabilitySet:
    """
    A collection of capabilities, with discovery and matching helpers.

    Usage:
        caps = CapabilitySet([
            AgentCapability("web-search", "Search the web"),
            AgentCapability("summarize",  "Summarize text"),
        ])

        caps.has("web-search")      # True
        caps.find("search")         # returns AgentCapability("web-search", ...)
    """

    def __init__(self, capabilities: Optional[List[AgentCapability]] = None):
        self._caps: Dict[str, AgentCapability] = {}
        for cap in (capabilities or []):
            self.add(cap)

    def add(self, capability: AgentCapability) -> "CapabilitySet":
        self._caps[capability.name] = capability
        return self

    def remove(self, name: str) -> "CapabilitySet":
        self._caps.pop(name, None)
        return self

    def has(self, name: str) -> bool:
        return name in self._caps

    def find(self, query: str) -> Optional[AgentCapability]:
        """Find a capability by name or tag."""
        for cap in self._caps.values():
            if cap.matches(query):
                return cap
        return None

    def find_all(self, query: str) -> List[AgentCapability]:
        return [c for c in self._caps.values() if c.matches(query)]

    def names(self) -> List[str]:
        return list(self._caps.keys())

    def to_list(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self._caps.values()]

    def __len__(self) -> int:
        return len(self._caps)

    def __iter__(self):
        return iter(self._caps.values())

    def __repr__(self) -> str:
        return f"CapabilitySet({self.names()})"


# ── Built-in well-known capabilities ─────────────────────────────────────────
# Standardized names encourage interoperability between frameworks.

WELL_KNOWN_CAPABILITIES = {
    "web-search": AgentCapability(
        name="web-search",
        description="Search the web for current information",
        tags=["search", "internet", "retrieval"],
    ),
    "summarize": AgentCapability(
        name="summarize",
        description="Summarize a document or text",
        tags=["nlp", "compression"],
    ),
    "code-execution": AgentCapability(
        name="code-execution",
        description="Execute code and return results",
        tags=["code", "python", "interpreter"],
    ),
    "rag-retrieval": AgentCapability(
        name="rag-retrieval",
        description="Retrieve documents from a knowledge base",
        tags=["rag", "retrieval", "knowledge"],
    ),
    "image-analysis": AgentCapability(
        name="image-analysis",
        description="Analyze and describe images",
        tags=["vision", "multimodal"],
    ),
    "planning": AgentCapability(
        name="planning",
        description="Break down complex tasks into steps",
        tags=["reasoning", "decomposition"],
    ),
    "memory": AgentCapability(
        name="memory",
        description="Store and retrieve long-term memory",
        tags=["storage", "recall"],
    ),
}
