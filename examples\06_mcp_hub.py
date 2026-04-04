"""
Example: MCP Hub - Multi-server MCP coordination.

This example demonstrates how to use the MCPHub to manage
multiple MCP servers and route tool calls across them.

Inspired by Skill_Seekers MCP patterns (26 tools support).

Usage:
    python examples/06_mcp_hub.py
"""

from agentlink.extensions.mcp_hub import MCPHub, MCPHubNode, MCPServerInfo


def example_hub_basic():
    """Basic MCP Hub usage."""
    print("=" * 60)
    print("Example: Basic MCP Hub")
    print("=" * 60)

    # Create hub
    hub = MCPHub()

    # Register servers (in real usage, these would be actual URLs)
    print("\n1. Registering MCP servers...")

    # Example: Web search MCP server
    hub.register_server(
        "search",
        "http://search-mcp:8080",
        tags=["web", "search", "research"],
        auto_discover=False,  # Skip auto-discover for demo
    )

    # Example: Code execution MCP server
    hub.register_server(
        "code",
        "http://code-mcp:8081",
        tags=["coding", "execution", "development"],
        auto_discover=False,
    )

    # Example: File system MCP server
    hub.register_server(
        "filesystem",
        "http://fs-mcp:8082",
        tags=["files", "storage", "io"],
        auto_discover=False,
    )

    print(f"   Registered servers: {list(hub.servers.keys())}")
    print(f"   Total tools indexed: {hub.tools_count}")

    # Simulate tool discovery
    print("\n2. Simulating tool discovery...")

    # Manually add tools for demo
    from agentlink.adapters.mcp import MCPTool

    search_server = hub.get_server("search")
    search_server.tools = [
        MCPTool(
            name="web_search",
            description="Search the web for information",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        ),
        MCPTool(
            name="get_page",
            description="Get content from a web page",
            parameters={"type": "object", "properties": {"url": {"type": "string"}}},
        ),
    ]

    code_server = hub.get_server("code")
    code_server.tools = [
        MCPTool(
            name="run_python",
            description="Execute Python code",
            parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        ),
        MCPTool(
            name="run_bash",
            description="Execute bash commands",
            parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        ),
    ]

    fs_server = hub.get_server("filesystem")
    fs_server.tools = [
        MCPTool(
            name="read_file",
            description="Read contents of a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    ]

    # Refresh index
    all_tools = hub.discover_all_tools()
    print(f"   Discovered {len(all_tools)} tools across all servers")

    print("\n3. Available tools:")
    for tool in all_tools:
        print(f"   - {tool['name']} (from: {tool['server']}, tags: {tool['tags']})")

    return hub


def example_hub_routing(hub: MCPHub):
    """Tool routing across servers."""
    print("\n" + "=" * 60)
    print("Example: Tool Routing")
    print("=" * 60)

    print("\n1. Finding tools by tag...")
    research_tools = hub.find_tools_by_tag("research")
    print(f"   Research tools: {[t['name'] for t in research_tools]}")

    coding_tools = hub.find_tools_by_tag("coding")
    print(f"   Coding tools: {[t['name'] for t in coding_tools]}")

    print("\n2. Finding specific tool...")
    tool_info = hub.find_tool("web_search")
    if tool_info:
        print(f"   Found: {tool_info['name']}")
        print(f"   Server: {tool_info['server']}")
        print(f"   Description: {tool_info['description']}")


def example_hub_node(hub: MCPHub):
    """Using MCPHubNode for agent integration."""
    print("\n" + "=" * 60)
    print("Example: MCPHubNode Integration")
    print("=" * 60)

    # Create a simple handler
    def my_handler(message):
        return f"Processed: {message}"

    # Create hub node
    node = MCPHubNode(
        "researcher-agent",
        handler=my_handler,
        mcp_hub=hub,
        auto_discover=False,
    )

    # Discover tools
    tools = node.discover_tools()

    print(f"\n1. Agent '{node.agent_id}' discovered {len(tools)} tools:")
    for tool in tools:
        print(f"   - {tool['name']}")

    print("\n2. Available tools property:")
    print(f"   {node.available_tools}")

    print("\n3. Note: In production, tool calls would be:")
    print("   result = node.call_mcp_tool('web_search', {'query': 'AI news'})")


def example_health_check(hub: MCPHub):
    """Health monitoring for MCP servers."""
    print("\n" + "=" * 60)
    print("Example: Health Monitoring")
    print("=" * 60)

    health = hub.health_check()

    print("\nServer Health Status:")
    for name, status in health.items():
        health_emoji = "✅" if status["healthy"] else "❌"
        tools_count = status["tools_count"]
        enabled = "enabled" if status["enabled"] else "disabled"

        print(f"   {health_emoji} {name}: {tools_count} tools ({enabled})")
        if "error" in status:
            print(f"      Error: {status['error']}")


if __name__ == "__main__":
    # Run examples
    hub = example_hub_basic()
    example_hub_routing(hub)
    example_hub_node(hub)
    example_health_check(hub)

    print("\n" + "=" * 60)
    print("✅ MCP Hub examples completed!")
    print("=" * 60)
    print("""
Production Usage Tips:
- Use real MCP server URLs in register_server()
- Enable auto_discover=True to fetch tools automatically
- Set timeout parameter for slow servers
- Use tags to categorize and filter tools
- Monitor server health with health_check() regularly
""")
