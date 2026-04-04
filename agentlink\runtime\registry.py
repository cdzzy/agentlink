"""
AgentRegistry — the discovery layer of AgentLink.

Agents announce themselves here so others can find them by ID,
capability, or namespace — without knowing each other's addresses.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentRecord:
    """A registration record for an agent in the registry."""
    agent_id: str
    namespace: str
    capabilities: List[str]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    _node_ref: Any = field(default=None, repr=False)   # weak ref to AgentNode

    def touch(self):
        self.last_seen = time.time()

    @property
    def address_str(self) -> str:
        return f"{self.agent_id}@{self.namespace}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "capabilities": self.capabilities,
            "description": self.description,
            "metadata": self.metadata,
            "registered_at": self.registered_at,
            "last_seen": self.last_seen,
        }


class AgentRegistry:
    """
    In-memory registry for agent discovery.

    Supports lookup by:
    - agent_id + namespace (exact)
    - capability name (semantic)
    - namespace (all agents in a group)

    Usage:
        registry = AgentRegistry()
        registry.register(node)

        # Find by capability
        nodes = registry.find_by_capability("web-search")

        # Find exact agent
        node = registry.find("researcher", namespace="production")
    """

    def __init__(self):
        self._records: Dict[str, AgentRecord] = {}   # key: "agent_id@namespace"
        self._lock = threading.RLock()

    def register(self, node: Any) -> AgentRecord:
        """Register an AgentNode in the registry."""
        key = f"{node.agent_id}@{node.namespace}"
        record = AgentRecord(
            agent_id=node.agent_id,
            namespace=node.namespace,
            capabilities=node.capabilities.names(),
            description=node.description,
            metadata=node.metadata,
            _node_ref=node,
        )
        with self._lock:
            self._records[key] = record
        return record

    def deregister(self, agent_id: str, namespace: str = "default"):
        """Remove an agent from the registry."""
        key = f"{agent_id}@{namespace}"
        with self._lock:
            self._records.pop(key, None)

    def find(self, agent_id: str, namespace: str = "default") -> Optional[Any]:
        """Find a specific AgentNode by ID and namespace."""
        key = f"{agent_id}@{namespace}"
        with self._lock:
            record = self._records.get(key)
            return record._node_ref if record else None

    def find_by_capability(self, capability: str, namespace: Optional[str] = None) -> List[Any]:
        """Find all agents that have a given capability."""
        result = []
        with self._lock:
            for record in self._records.values():
                if namespace and record.namespace != namespace:
                    continue
                if capability in record.capabilities:
                    result.append(record._node_ref)
        return result

    def find_by_namespace(self, namespace: str) -> List[Any]:
        """Find all agents in a namespace."""
        with self._lock:
            return [
                r._node_ref for r in self._records.values()
                if r.namespace == namespace and r._node_ref is not None
            ]

    def all_agents(self) -> List[AgentRecord]:
        with self._lock:
            return list(self._records.values())

    def touch(self, agent_id: str, namespace: str = "default"):
        key = f"{agent_id}@{namespace}"
        with self._lock:
            if key in self._records:
                self._records[key].touch()

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            namespaces: Dict[str, int] = {}
            capabilities: Dict[str, int] = {}
            for r in self._records.values():
                namespaces[r.namespace] = namespaces.get(r.namespace, 0) + 1
                for cap in r.capabilities:
                    capabilities[cap] = capabilities.get(cap, 0) + 1
            return {
                "total_agents": len(self._records),
                "namespaces": namespaces,
                "capabilities": capabilities,
            }

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"AgentRegistry({len(self._records)} agents)"
