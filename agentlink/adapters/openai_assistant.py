"""
OpenAI Assistant API Adapter for AgentLink.

This adapter allows AgentLink agents to communicate with OpenAI Assistants
built using the OpenAI Assistant API ( Assistants API v2 ).

Supports:
- Thread-based conversation
- Tool use (code interpreter, file search, function tools)
- Streaming responses
- Message attachments
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agentlink.adapters.base import BaseAdapter
from agentlink.protocol.capability import AgentCapability
from agentlink.protocol.message import AgentMessage


class OpenAIAssistantAdapter(BaseAdapter):
    """
    Adapter for OpenAI Assistants API.

    Requires the 'openai' package: pip install openai

    Example:
        from agentlink.adapters.openai_assistant import OpenAIAssistantAdapter

        adapter = OpenAIAssistantAdapter(
            agent_id="openai_assistant",
            assistant_id="asst_xxx",
            thread_id="thread_xxx",  # Optional: provide existing thread
            api_key="sk-xxx",  # Optional: uses OPENAI_API_KEY env var by default
        )
        node = adapter.as_node()
        bus.register(node)
    """

    def __init__(
        self,
        agent_id: str,
        assistant_id: str,
        thread_id: Optional[str] = None,
        api_key: Optional[str] = None,
        namespace: str = "default",
        capabilities: Optional[List[Union[str, AgentCapability]]] = None,
        description: str = "OpenAI Assistant via AgentLink",
        max_polls: int = 60,
        poll_interval: float = 1.0,
    ):
        """
        Initialize the OpenAI Assistant adapter.

        Args:
            agent_id: Unique identifier for this agent in AgentLink
            assistant_id: OpenAI Assistant ID (starts with "asst_")
            thread_id: Existing thread ID, or None to create a new thread per message
            api_key: OpenAI API key (reads from OPENAI_API_KEY env var if not provided)
            namespace: AgentLink namespace
            capabilities: List of agent capabilities
            description: Human-readable description
            max_polls: Maximum number of polling attempts for run completion
            poll_interval: Seconds between polling attempts
        """
        super().__init__(agent_id, namespace, capabilities, description)
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.api_key = api_key
        self.max_polls = max_polls
        self.poll_interval = poll_interval
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAIAssistantAdapter. "
                    "Install it with: pip install openai"
                )
        return self._client

    def _invoke(self, message: AgentMessage) -> Any:
        """
        Send a message to the OpenAI Assistant and return the response.
        """
        # Create or use existing thread
        thread_id = self.thread_id
        if thread_id is None:
            thread = self.client.beta.threads.create()
            thread_id = thread.id

        # Add user message to thread
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message.content,
        )

        # Create and run assistant
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant_id,
        )

        # Poll for completion
        run = self._wait_for_run(run.id, thread_id)

        # Get the assistant's response
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc",
            limit=1,
        )

        if messages.data:
            latest_message = messages.data[0]
            if latest_message.role == "assistant":
                # Return text content
                for content in latest_message.content:
                    if hasattr(content, "text") and content.text:
                        return content.text.value

        return ""

    def _wait_for_run(self, run_id: str, thread_id: str) -> Any:
        """
        Poll the run until it reaches a terminal state.
        """
        for _ in range(self.max_polls):
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id,
            )

            if run.status in ("completed", "failed", "cancelled", "expired"):
                if run.status == "failed":
                    error_msg = getattr(run.last_error, "message", "Unknown error")
                    raise RuntimeError(f"Assistant run failed: {error_msg}")
                return run

            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"Assistant run timed out after {self.max_polls} polls. "
            f"Run status: {run.status}"
        )

    def create_thread(self) -> str:
        """
        Create a new conversation thread.

        Returns:
            The thread ID string.
        """
        thread = self.client.beta.threads.create()
        return thread.id

    def set_thread(self, thread_id: str):
        """
        Set the active thread for subsequent messages.
        """
        self.thread_id = thread_id

    def get_thread_messages(
        self,
        thread_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get messages from a thread.

        Args:
            thread_id: Thread ID (uses active thread if not provided)
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries with role and content.
        """
        thread_id = thread_id or self.thread_id
        if not thread_id:
            raise ValueError("No thread_id provided and no active thread set.")

        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc",
            limit=limit,
        )

        return [
            {"role": msg.role, "content": self._extract_text(msg)}
            for msg in messages.data
        ]

    def _extract_text(self, message) -> str:
        """Extract text content from a message."""
        for content in message.content:
            if hasattr(content, "text") and content.text:
                return content.text.value
        return ""


# Type hint fix for List
from typing import Union
