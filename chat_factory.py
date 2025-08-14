import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_groq import ChatGroq
from agent import Agent
from dotenv import load_dotenv

from mcp_agent_v2 import MyMCPAgent

load_dotenv()

def get_model(name: str, **kwargs) -> BaseLLM:
    if 'gemini' in name:
        return ChatGoogleGenerativeAI(model=name, safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },**kwargs)
    elif name.startswith('gpt'):
        return ChatOpenAI(model_name=name)
    else:
        kwargs.pop('encoded_image')
        return ChatGroq(model=name,
                        max_retries=2,
                        **kwargs)


def chat(query, model='gemini', type='chat', **kwargs):
    retriever = kwargs.pop('retriever')
    model = get_model(model, **kwargs)
    if type == 'agent':
        agent = Agent(model)
        resp = agent.chat(query)
        return resp.content
    elif type == 'chat':
        from chatbot import chat
        return chat(query, model, kwargs.get('encoded_image'))
    elif type == 'rag':
        from rag import RAG
        rag = RAG(model, retriever)
        return rag.chat(query)
    elif type == 'research_assistant':
        from research_assistant import ResearchAssistant
        rag = ResearchAssistant(model)
        return rag.chat(query)
    # elif type == 'mcp_agent':
    #     from mcp_agent_v2 import MyMCPAgent
    #     agent = MyMCPAgent(model)
    #     # from mcp_agent import MCPAgent
    #     # agent = MCPAgent(model)
    #     # return agent.chat(query)
    #     return agent.chat(query)



if __name__ == '__main__':
    agent = MyMCPAgent(get_model('gemini-2.5-flash'))
    print(asyncio.run(agent.chat("hello, add 5 and 9")))