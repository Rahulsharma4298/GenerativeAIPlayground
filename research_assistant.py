from operator import add
from typing import TypedDict, Annotated, List

from langchain_community.document_loaders import WikipediaLoader
from langchain_core.language_models import BaseChatModel
from langgraph.graph import START, END, StateGraph
from langchain_community.tools import TavilySearchResults


class ResearchAssistant:
    ROOT_NODE = "ResearchAssistant"

    def __init__(self, model: BaseChatModel):
        self.model = model

    class State(TypedDict):
        topic: str
        context: Annotated[List[str], add]
        report: str

    @staticmethod
    def search_tavily(state: State):
        tavily_search_tool = TavilySearchResults(
            max_results=6,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            verbose=True
        )
        results = tavily_search_tool.run(state['topic'])
        print(results)
        return {"context": [results]}

    @staticmethod
    def search_wikipedia(state: State):
        wp_loader = WikipediaLoader(load_max_docs=5, query=state['topic'])
        documents = wp_loader.load()
        formatted_data = [{"context": document.page_content,
                           "source": document.metadata['source']} for document in documents]
        return {"context": formatted_data}

    def research_assistant(self, state: State):
        prompt = ("You are an expert research analyst."
                  "Based on given context, generate a good detailed report"
                  "which includes detailed points about each point."
                  "Format the output as Markdown."
                  "Always includes sources section at last."
                  "<context>{context}</context>")
        response = self.model.invoke(prompt.format(context=state['context']))
        return {"report": response}


    def build_graph(self):
        builder = StateGraph(ResearchAssistant.State)
        builder.add_node("search_tavily", ResearchAssistant.search_tavily)
        builder.add_node("search_wikipedia", ResearchAssistant.search_wikipedia)
        builder.add_node(ResearchAssistant.ROOT_NODE, self.research_assistant)
        builder.add_edge(START, "search_tavily")
        builder.add_edge(START, "search_wikipedia")
        builder.add_edge("search_tavily", ResearchAssistant.ROOT_NODE)
        builder.add_edge("search_wikipedia", ResearchAssistant.ROOT_NODE)
        builder.add_edge(ResearchAssistant.ROOT_NODE, END)
        graph = builder.compile()
        return graph

    def chat(self, query):
        return self.build_graph().invoke({"topic": query})['report'].content

