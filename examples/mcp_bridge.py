"""
agentlink examples: MCP Protocol Bridge
参考 huggingface/skills 的工具生态集成思路，
为 agentlink 添加 MCP (Model Context Protocol) 兼容桥接能力，
使 agent 间通信可以接入更广泛的工具生态。
"""

import asyncio
import json
from typing import Any


class MCPToolAdapter:
    """
    MCP 工具适配器：将 MCP 工具调用转换为 agentlink 消息格式。
    这使得 agentlink 可以作为 MCP 生态与多 Agent 系统之间的桥梁，
    参考 huggingface/skills "Give your agents the power of the Hugging Face ecosystem"。
    """

    def __init__(self, bus):
        self.bus = bus
        self._tool_registry: dict[str, dict] = {}

    def register_tool(self, name: str, description: str, schema: dict):
        """注册一个 MCP 工具到本地注册表"""
        self._tool_registry[name] = {
            "name": name,
            "description": description,
            "input_schema": schema,
            "registered_at": asyncio.get_event_loop().time(),
        }

    async def invoke_tool(self, tool_name: str, params: dict) -> Any:
        """通过 agentlink 总线调用注册的 MCP 工具"""
        if tool_name not in self._tool_registry:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        tool = self._tool_registry[tool_name]
        # 构建 MCP 格式的调用请求
        mcp_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params,
            },
        }

        # 通过 agentlink 发送工具调用
        response = await self.bus.publish(
            channel=f"mcp://tools/{tool_name}",
            payload=mcp_request,
        )
        return response

    def list_tools(self) -> list[dict]:
        """返回所有已注册工具的描述"""
        return [
            {"name": t["name"], "description": t["description"]}
            for t in self._tool_registry.values()
        ]


class AgentLinkMCPBridge:
    """
    AgentLink <-> MCP 协议桥接器。
    将 MCP 的工具/资源/提示模板协议映射到 agentlink 的消息总线架构。
    核心参考：huggingface/skills 的跨生态工具集成理念。
    """

    def __init__(self):
        self.tool_adapter = MCPToolAdapter(bus=None)  # bus 在实际初始化时注入
        self._resource_cache: dict[str, Any] = {}

    def mcp_resource_to_agentlink(self, resource_uri: str) -> dict:
        """
        将 MCP 资源转换为 agentlink 可消费的标准化消息格式。
        """
        # 从缓存或远程获取资源
        resource_data = self._resource_cache.get(
            resource_uri,
            self._fetch_mcp_resource(resource_uri)
        )

        return {
            "channel": "resource",
            "uri": resource_uri,
            "content": resource_data,
            "content_type": self._infer_content_type(resource_uri),
        }

    def _fetch_mcp_resource(self, uri: str) -> Any:
        """实际 MCP 资源获取逻辑（占位）"""
        return {"uri": uri, "data": "placeholder"}

    def _infer_content_type(self, uri: str) -> str:
        """根据 URI 推断资源类型"""
        if uri.endswith(".json"):
            return "application/json"
        if uri.endswith(".md"):
            return "text/markdown"
        return "text/plain"

    async def route_agent_message(self, agent_id: str, message: dict) -> dict:
        """
        路由 Agent 消息，注入 MCP 工具调用上下文。
        用于 Multica 风格的多 Agent 协作场景。
        """
        tools = self.tool_adapter.list_tools()
        message["available_tools"] = tools
        return message


# 使用示例
async def demo():
    bridge = AgentLinkMCPBridge()

    # 注册一些 MCP 工具（模拟 huggingface/skills 风格）
    bridge.tool_adapter.register_tool(
        name="huggingface_model_inference",
        description="Run inference on Hugging Face models",
        schema={
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "input": {"type": "string"},
            },
            "required": ["model", "input"],
        },
    )

    bridge.tool_adapter.register_tool(
        name="search_datasets",
        description="Search Hugging Face datasets",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
        },
    )

    print("已注册 MCP 工具:", bridge.tool_adapter.list_tools())


if __name__ == "__main__":
    asyncio.run(demo())
