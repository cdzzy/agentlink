"""
Agent memory adapter for AgentLink.

Enables agents to store, recall, and share memories across sessions and
across agents via the message bus. Inspired by mem0's cross-session memory
pattern and hermes-agent's self-improvement memory loop.

Features:
- Per-agent session memory with automatic key extraction
- Shared memory namespace for multi-agent collaboration
- Memory recall with relevance scoring
- Memory-based prompt enrichment (inject relevant memories into messages)
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from agentlink.protocol.message import AgentMessage


@dataclass
class Memory:
    """A single unit of agent memory."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    agent_id: str = ""
    namespace: str = "default"
    content: str = ""
    tags: List[str] = field(default_factory=list)
    source_message_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "content": self.content,
            "tags": self.tags,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "importance": round(self.importance, 3),
        }


@dataclass
class RecallResult:
    """Result of a memory recall operation."""
    memories: List[Memory]
    query: str
    total_matched: int = 0

    def to_prompt_context(self, max_chars: int = 2000) -> str:
        """Format recalled memories as a prompt context string."""
        if not self.memories:
            return ""
        lines = ["[Relevant memories from previous sessions:]"]
        total = 0
        for m in self.memories:
            line = f"- [{', '.join(m.tags)}] {m.content}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)


class AgentMemory:
    """
    In-memory store for agent memories with recall capabilities.

    Usage::

        memory = AgentMemory(agent_id="researcher")

        # Store a memory
        memory.store(
            content="User prefers concise responses with code examples",
            tags=["preference", "style"],
            importance=0.8,
        )

        # Recall relevant memories
        results = memory.recall("How should I respond to the user?")
        if results.memories:
            context = results.to_prompt_context()
            print(context)  # Inject into prompt
    """

    def __init__(
        self,
        agent_id: str,
        namespace: str = "default",
        max_memories: int = 10000,
        decay_interval: float = 3600.0,  # seconds
    ):
        self.agent_id = agent_id
        self.namespace = namespace
        self.max_memories = max_memories
        self.decay_interval = decay_interval
        self._memories: Dict[str, Memory] = {}
        self._tag_index: Dict[str, Set[str]] = {}

    def store(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        source_message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """
        Store a new memory.

        Args:
            content: The memory content (natural language)
            tags: List of tags for categorization and retrieval
            importance: Importance score (0.0-1.0), affects decay rate
            source_message_id: Optional link to the originating message
            metadata: Additional metadata

        Returns:
            The created Memory object.
        """
        memory = Memory(
            agent_id=self.agent_id,
            namespace=self.namespace,
            content=content,
            tags=tags or [],
            source_message_id=source_message_id,
            importance=max(0.0, min(1.0, importance)),
            metadata=metadata or {},
        )

        self._memories[memory.id] = memory

        # Update tag index
        for tag in memory.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = set()
            self._tag_index[tag_lower].add(memory.id)

        # Evict oldest if over capacity
        if len(self._memories) > self.max_memories:
            self._evict_oldest()

        return memory

    def recall(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        limit: int = 5,
    ) -> RecallResult:
        """
        Recall memories relevant to a query.

        Uses keyword matching and tag filtering for relevance scoring.
        For production use, replace with embedding-based semantic search.

        Args:
            query: The query text to match against
            tags: Optional tag filter (any match)
            min_importance: Minimum importance threshold
            limit: Maximum number of results

        Returns:
            RecallResult with matched memories sorted by relevance.
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        candidates: List[Tuple[Memory, float]] = []

        for memory in self._memories.values():
            # Filter by importance
            if memory.importance < min_importance:
                continue

            # Filter by tags
            if tags:
                memory_tags_lower = {t.lower() for t in memory.tags}
                if not memory_tags_lower.intersection({t.lower() for t in tags}):
                    continue

            # Score: keyword overlap
            memory_terms = set(memory.content.lower().split())
            overlap = len(query_terms & memory_terms)
            if overlap == 0 and query_lower not in memory.content.lower():
                continue

            score = overlap / max(len(query_terms), 1)
            # Boost by importance and recency
            score *= (0.5 + 0.3 * memory.importance + 0.2 * self._recency_factor(memory))

            candidates.append((memory, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        results = candidates[:limit]

        # Update access stats
        for memory, _ in results:
            memory.last_accessed_at = time.time()
            memory.access_count += 1

        return RecallResult(
            memories=[m for m, _ in results],
            query=query,
            total_matched=len(candidates),
        )

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        return self._memories.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        memory = self._memories.pop(memory_id, None)
        if memory:
            for tag in memory.tags:
                tag_lower = tag.lower()
                if tag_lower in self._tag_index:
                    self._tag_index[tag_lower].discard(memory_id)
            return True
        return False

    def by_tag(self, tag: str) -> List[Memory]:
        """Get all memories with a specific tag."""
        ids = self._tag_index.get(tag.lower(), set())
        return [self._memories[mid] for mid in ids if mid in self._memories]

    def all_memories(self) -> List[Memory]:
        """Get all memories."""
        return list(self._memories.values())

    def count(self) -> int:
        """Return total number of stored memories."""
        return len(self._memories)

    def stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        if not self._memories:
            return {"total": 0, "tags": 0}

        memories = list(self._memories.values())
        return {
            "total": len(memories),
            "tags": len(self._tag_index),
            "avg_importance": round(
                sum(m.importance for m in memories) / len(memories), 3
            ),
            "most_accessed": max(
                ((m.access_count, m.content[:50]) for m in memories),
                default=(0, ""),
            ),
            "top_tags": sorted(
                self._tag_index.keys(),
                key=lambda t: len(self._tag_index[t]),
                reverse=True,
            )[:10],
        }

    def _recency_factor(self, memory: Memory) -> float:
        """Calculate recency factor (0-1, decays over time)."""
        age_seconds = time.time() - memory.created_at
        half_life = self.decay_interval * 24  # 24 decay intervals to halve
        import math
        return math.exp(-0.693 * age_seconds / half_life)

    def _evict_oldest(self) -> None:
        """Evict the lowest-scoring memory when at capacity."""
        if not self._memories:
            return
        # Score = importance * recency
        def eviction_score(m: Memory) -> float:
            return m.importance * (1.0 - self._recency_factor(m))

        weakest = min(self._memories.values(), key=eviction_score)
        self.delete(weakest.id)


class SharedMemorySpace:
    """
    Shared memory space for multi-agent collaboration.

    Allows multiple agents to read/write to a shared memory namespace,
    enabling cross-agent knowledge sharing.

    Usage::

        shared = SharedMemorySpace(namespace="team_knowledge")
        shared.store("researcher", "API rate limit is 100 req/min")
        shared.store("writer", "User prefers formal tone")

        # Any agent can recall from shared space
        results = shared.recall("What is the API rate limit?")
    """

    def __init__(self, namespace: str = "shared", max_memories: int = 50000):
        self.namespace = namespace
        self._agent_memories: Dict[str, AgentMemory] = {}
        self._shared_store = AgentMemory(
            agent_id="shared",
            namespace=namespace,
            max_memories=max_memories,
        )

    def store(
        self,
        agent_id: str,
        content: str,
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        shared: bool = False,
    ) -> Memory:
        """Store a memory, optionally sharing it with all agents."""
        memory = self._get_agent_memory(agent_id).store(
            content=content,
            tags=tags,
            importance=importance,
        )
        if shared:
            self._shared_store.store(
                content=f"[{agent_id}] {content}",
                tags=tags,
                importance=importance,
            )
        return memory

    def recall(
        self,
        query: str,
        agent_id: Optional[str] = None,
        include_shared: bool = True,
        **kwargs,
    ) -> RecallResult:
        """Recall memories from agent and optionally shared space."""
        all_memories: List[Memory] = []

        if agent_id:
            agent_result = self._get_agent_memory(agent_id).recall(query, **kwargs)
            all_memories.extend(agent_result.memories)

        if include_shared:
            shared_result = self._shared_store.recall(query, **kwargs)
            all_memories.extend(shared_result.memories)

        # Deduplicate by content hash
        seen: Set[str] = set()
        unique: List[Memory] = []
        for m in all_memories:
            h = hashlib.md5(m.content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(m)

        return RecallResult(
            memories=unique[:kwargs.get("limit", 5)],
            query=query,
            total_matched=len(all_memories),
        )

    def _get_agent_memory(self, agent_id: str) -> AgentMemory:
        if agent_id not in self._agent_memories:
            self._agent_memories[agent_id] = AgentMemory(
                agent_id=agent_id,
                namespace=self.namespace,
            )
        return self._agent_memories[agent_id]

    def agent_count(self) -> int:
        return len(self._agent_memories)

    def total_memories(self) -> int:
        return sum(m.count() for m in self._agent_memories.values()) + self._shared_store.count()
