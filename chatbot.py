from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

messages = [SystemMessage("You are a helpful assistant"),
            MessagesPlaceholder('chat_history'),
            MessagesPlaceholder('messages'),
            ]

def get_chain(model):
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | model
    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
        history_messages_key="chat_history",
    )
    return chain_with_message_history

# In-memory store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """ Function to get chat history from store """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



config = {"configurable": {"session_id": "abc2"}}

def chat(query, model):
    return get_chain(model).stream({"messages": [HumanMessage(query)]},
                                             config=config)
