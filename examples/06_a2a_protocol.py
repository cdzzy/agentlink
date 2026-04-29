"""
A2A Protocol Integration Example — AgentLink as an A2A server and client.

This example shows two ways to integrate with the A2A (Agent-to-Agent) protocol:
1. A2AServerAdapter: expose an AgentLink node as an A2A HTTP endpoint
2. A2AAdapter: connect to a remote A2A agent and call it from AgentLink

Reference: https://a2a-protocol.org
"""

import asyncio


async def example_server():
    """
    Example 1: Expose an AgentLink node as an A2A server.

    The A2AServerAdapter:
    - Publishes an Agent Card at /.well-known/agent.json
    - Handles A2A JSON-RPC requests at POST /a2a
    - Translates A2A messages to AgentLink messages and calls the node
    """
    from agentlink import AgentNode, AgentBus, AgentMessage, MessageType
    from agentlink.adapters.a2a_adapter import A2AServerAdapter, A2AAgentCard

    # Define a simple agent
    def researcher(message: AgentMessage) -> str:
        return f"Research complete for: {message.content}"

    # Create AgentLink node
    node = AgentNode(
        agent_id="researcher",
        handler=researcher,
        capabilities=["web-search"],
        description="Research agent for market data",
    )

    # Wrap as A2A server
    server = A2AServerAdapter(
        node=node,
        host="0.0.0.0",
        port=8000,
        agent_card=A2AAgentCard(
            name="researcher",
            description="Research agent for market data analysis",
            skills=[
                {"id": "web-search", "name": "Web Search"},
                {"id": "data-analysis", "name": "Data Analysis"},
            ],
            endpoint="http://localhost:8000",
        ),
    )

    print("Starting A2A server...")
    print("Agent Card available at: http://localhost:8000/.well-known/agent.json")
    print("A2A JSON-RPC endpoint: http://localhost:8000/a2a")

    # Run the server
    # await server.start()  # Uncomment to run
    print("(Server example complete — uncomment server.start() to run)")


async def example_client():
    """
    Example 2: Call a remote A2A agent from AgentLink.

    A2AAdapter wraps a remote A2A endpoint as an AgentLink AgentNode,
    so it can be registered on any AgentLink bus.
    """
    from agentlink import AgentBus
    from agentlink.adapters.a2a_adapter import A2AAdapter

    # Option A: Direct connection
    remote = A2AAdapter(
        agent_url="http://analyst-agent:8000",
        agent_id="remote-analyst",
        capabilities=["data-analysis"],
        description="Remote A2A analyst agent",
    )
    node = remote.as_node()

    # Option B: Auto-discover from Agent Card
    # card = await A2AAdapter.get_agent_card("http://analyst-agent:8000")
    # remote = A2AAdapter(agent_url="http://analyst-agent:8000", agent_id=card.name)
    # node = remote.as_node()

    bus = AgentBus()
    bus.register(node)

    result = node.send("remote-analyst", "What were Q1 sales?")
    print(f"Remote A2A result: {result.content}")


def example_agent_card():
    """Example 3: Generate an A2A Agent Card from AgentLink metadata."""
    from agentlink.adapters.a2a_adapter import A2AAgentCard
    import json

    card = A2AAgentCard(
        name="cdzzy-researcher",
        description="Multi-domain research agent powered by AgentLink",
        version="1.0.0",
        skills=[
            {"id": "web-search", "name": "Web Search"},
            {"id": "data-analysis", "name": "Data Analysis"},
            {"id": "code-execution", "name": "Code Execution"},
        ],
        tags=["research", "ai", "agent"],
        provider={"organization": "cdzzy", "name": "AgentLink"},
        url="https://github.com/cdzzy/agentlink",
    )

    print("A2A Agent Card:")
    print(json.dumps(card.to_dict(), indent=2))


if __name__ == "__main__":
    import sys

    print("=== A2A Protocol Integration Examples ===\n")

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        asyncio.run(example_server())
    elif len(sys.argv) > 1 and sys.argv[1] == "client":
        asyncio.run(example_client())
    else:
        example_agent_card()
