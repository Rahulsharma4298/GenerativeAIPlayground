import requests
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools import TavilySearchResults, DuckDuckGoSearchResults, YouTubeSearchTool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import tool

from med_rag_test.tools import search_medicine

store = {}


@tool
def get_recipes(query):
    """ Returns recipe for given query"""
    url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={query}"
    resp = requests.get(url)
    return resp.json()['meals']


class Agent:
    def __init__(self, model: BaseLLM):
        tools = Agent.get_tools()
        self.model = model
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools,
                                       verbose=True, handle_parsing_errors=True)

        self._agent = RunnableWithMessageHistory(
            agent_executor,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    @staticmethod
    def get_tools():
        search = TavilySearchResults(max_results=5,
                                     search_depth="advanced",
                                     include_answer=True,
                                     include_raw_content=False,
                                     description="For given query, please do search")
        ddg_search = DuckDuckGoSearchResults()
        tools = [ddg_search,
                 YouTubeSearchTool(description="Return youtube video link, only 1 video and only link"),
                 get_recipes,
                 search_medicine]
        return tools

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    def chat(self, query):
        return self._agent.invoke({"input": query},
                                  self.config)['output']

    @property
    def config(self):
        config = {"configurable": {"session_id": "abc123"}}
        return config
