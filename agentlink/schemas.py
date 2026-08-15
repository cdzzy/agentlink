"""
Structured message schemas with runtime validation (Issue #1).

AgentLink messages are untyped by default. This module adds opt-in schema
validation so integration bugs (e.g. ``user_id`` vs ``userId``) are caught at
the boundary rather than deep in agent logic.

Schemas can be Pydantic models, dataclasses, or any class with a constructor
accepting keyword arguments from a dict.

Usage::

    from agentlink.schemas import MessageSchema, SchemaRegistry
    from pydantic import BaseModel

    class TaskMessage(MessageSchema, BaseModel):
        task_id: str
        priority: int = 1
        payload: dict

    registry = SchemaRegistry()
    registry.register("task", TaskMessage)

    validated = registry.validate("task", {"task_id": "t-001", "payload": {}})
    # validated is a TaskMessage instance
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type


class MessageSchema:
    """
    Base class for typed message schemas.

    Subclass alongside a validation library (Pydantic) or as a plain dataclass::

        from pydantic import BaseModel
        class TaskMessage(MessageSchema, BaseModel):
            task_id: str

        from dataclasses import dataclass
        @dataclass
        class ResultMessage(MessageSchema):
            data: dict
    """

    @classmethod
    def coerce(cls, content: Any) -> Any:
        """
        Validate/coerce ``content`` into an instance of this schema.

        Uses Pydantic's ``model_validate`` (v2) or ``parse_obj`` (v1) when
        available, otherwise constructs via ``cls(**content)``.

        Raises:
            TypeError: If content cannot be coerced into this schema.
        """
        if hasattr(cls, "model_validate"):          # Pydantic v2
            return cls.model_validate(content)
        if hasattr(cls, "parse_obj"):               # Pydantic v1
            return cls.parse_obj(content)
        if isinstance(content, cls):
            return content
        if isinstance(content, dict):
            return cls(**content)
        raise TypeError(
            f"Cannot coerce {type(content).__name__} into {cls.__name__}"
        )


class SchemaRegistry:
    """
    Registry mapping message type names to schema classes.

    Attach one to an ``AgentBus`` (or use standalone) to validate message
    payloads before they are routed.
    """

    def __init__(self) -> None:
        self._schemas: Dict[str, Type[MessageSchema]] = {}

    def register(self, name: str, schema_cls: Type[MessageSchema]) -> "SchemaRegistry":
        """
        Register a schema under a message type name.

        Args:
            name: Logical message type (e.g. ``"task"``).
            schema_cls: A MessageSchema subclass.

        Returns:
            self (for chaining).
        """
        self._schemas[name] = schema_cls
        return self

    def get(self, name: str) -> Optional[Type[MessageSchema]]:
        """Return the schema registered for ``name``, or None."""
        return self._schemas.get(name)

    def validate(self, name: str, content: Any) -> Any:
        """
        Validate ``content`` against the schema registered as ``name``.

        Returns:
            The validated/coerced schema instance.

        Raises:
            ValueError: If no schema is registered under ``name``.
        """
        schema_cls = self._schemas.get(name)
        if schema_cls is None:
            raise ValueError(
                f"No schema registered for message type {name!r}. "
                f"Registered: {self.names()}"
            )
        return schema_cls.coerce(content)

    def names(self) -> List[str]:
        """Return the registered message type names."""
        return list(self._schemas.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._schemas

    def __len__(self) -> int:
        return len(self._schemas)
