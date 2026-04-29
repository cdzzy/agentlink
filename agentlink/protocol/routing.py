"""
AgentLink Routing — how messages get from A to B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RoutingStrategy(str, Enum):
    DIRECT      = "direct"       # Send to a specific agent by ID
    CAPABILITY  = "capability"   # Send to any agent with matching capability
    BROADCAST   = "broadcast"    # Send to all agents in a namespace
    ROUND_ROBIN = "round_robin"  # Distribute across capable agents
    FIRST_MATCH = "first_match"  # Send to first available capable agent


@dataclass
class RoutingRule:
    """
    A routing rule that maps message patterns to destinations.

    Used by the AgentBus to make routing decisions.
    """
    strategy: RoutingStrategy = RoutingStrategy.DIRECT
    capability_filter: Optional[str] = None    # route to agents with this capability
    namespace_filter: Optional[str] = None     # restrict to this namespace
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0                          # higher = checked first

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "capability_filter": self.capability_filter,
            "namespace_filter": self.namespace_filter,
            "metadata_filters": self.metadata_filters,
            "priority": self.priority,
        }
