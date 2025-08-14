from langchain_core.language_models import BaseLLM
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPAgent:
    def __init__(self, model: BaseLLM):
        """
        Initializes the agent's configuration.
        Note: The agent itself is created asynchronously later.
        """
        self.model = model
        self.server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@notionhq/notion-mcp-server"],
            env={"OPENAPI_MCP_HEADERS": "{\"Authorization\": \"Bearer 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\", \"Notion-Version\": \"2022-06-28\" }"}
        )
        # The agent will be created on the first call to chat()
        self._agent = None

    async def _get_mcp_server_tools(self):
        """Asynchronously connects to the MCP server and loads tools."""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
        print("Available tools:", *[tool.name for tool in tools])
        return tools

    async def _create_agent(self):
        """Asynchronously creates the agent with the loaded tools."""
        tools = await self._get_mcp_server_tools()
        agent_executor = create_react_agent(self.model, tools)
        return agent_executor

    async def chat(self, message: str):
        """
        Handles a chat message by ensuring the agent is initialized and then invoking it.
        """
        if self._agent is None:
            print("First time setup: Initializing agent...")
            self._agent = await self._create_agent()
            print("Agent initialized.")

        input_data = {"messages": [("user", message)]}
        return self._agent.invoke(input_data, output_keys='messages')[-1].content
