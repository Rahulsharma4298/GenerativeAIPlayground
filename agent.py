import uuid

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults, YouTubeSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseLLM
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import yfinance
from med_rag_test.tools import search_medicine

load_dotenv()




class Agent:
    memory = MemorySaver()
    def __init__(self, model: BaseLLM):
        self.thread_id = str(uuid.uuid1())
        tools = Agent.get_tools()
        self.memory = MemorySaver()
        self.model = model.bind_tools(tools)
        self._agent_executor = create_react_agent(self.model,
                                                  tools,
                                                  checkpointer=self.memory,
                                                  debug=True,
                                                  messages_modifier='Do not use tool when user makes general conversations.'
                                                                    'Format the response in Markdown.'
                                                                    'if the response has image, always render it as Markdown.'
                                                                    'If the response has youtube video url, return only its url without any text.')

    @staticmethod
    @tool
    def tavily_search(query: str):
        """
        Use this tool to perform a advance web search on internet using Tavily.

        User can search ANY query like weather, news, general search, answers, finance etc

        Properly use the content and show news in bullet points.

        Don't answer no or ask user to search, instead just search it.

        User may ask n number of follow-up questions, do search it as well.

        Do not assume and answer at your own, first search it then answer.

        Do not say 'I cannot access real-time and all' before searching.

        :param query: The search query.
        """
        tavily_search_tool = TavilySearchResults(
            max_results=6,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False
        )
        # Perform the search and return the results
        return tavily_search_tool.run(query)

    @staticmethod
    @tool
    def duckduckgo_search(query: str):
        """
        Use this tool to perform a quick web search on internet using DuckDuckGo.

        Don't answer no or ask user to search, instead just search it.

        User may ask n number of follow-up questions, do search it as well.

        Do not assume and answer at your own, first search it then answer.

        Do not say 'I cannot access real-time and all' before searching.

        :param query: The search query.
        """
        ddg_search_tool = DuckDuckGoSearchResults()
        return ddg_search_tool.run(query)


    @tool
    @staticmethod
    def yfinance_search(query: str):
        """ Search for anything related to finance, stocks and related news
        :param query: The search query.
        """
        yf = yfinance.Ticker(query)
        print(yf.get_info())
        return yf.get_info()


    @staticmethod
    def get_tools():
        yts = YouTubeSearchTool(description="Return only 1 video url. "
                                               "No explanation or any text other than url")
        tools = [Agent.tavily_search, Agent.duckduckgo_search, yts, search_medicine, Agent.yfinance_search]
        return tools

    def get_executor(self):
        return self._agent_executor

    def chat(self, query):
        executor = self.get_executor()
        return executor.invoke({"messages": [HumanMessage(query)]},
                               self.config,
                               output_keys='messages')[-1]

    @property
    def config(self):
        config = {"configurable": {"thread_id": self.thread_id}}
        print(config)
        return config


# if __name__ == '__main__':
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#     agent = Agent(model)
#     executor = agent.get_executor()
#     resp = executor.invoke({"messages": [HumanMessage('latest news india')]},
#                            agent.config, output_keys='messages')
#     print(resp[-1].pretty_print())
