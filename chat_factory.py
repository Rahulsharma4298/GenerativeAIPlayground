from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from agent import Agent
from dotenv import load_dotenv

load_dotenv()

def get_model(name: str, **kwargs) -> BaseLLM:
    if 'gemini' in name:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", **kwargs)
    elif 'gpt' in name:
        return ChatOpenAI(model_name=name)
    else:
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



