from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from med_rag_test.retriever import get_retriever

model = ChatGroq(model="llama3-70b-8192", max_retries=2)
template = ("You are a helpful pharmacy assistant."
            "Based on user symptoms & queries suggest the medicines."
            "Provides medicine information for educational purpose."
            "Use the context for answering."
            "You may ask for more symptoms if required."
            "Keep the answer short & concise."
            "Only search the context if user tells symptoms not for any other things like '3 months'"
            "Do not suggest anything at your own, just use the context."
            "question: {question}"
            "context: {context}")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model | StrOutputParser()

def chat(query):
    context = get_retriever().invoke(query)
    print(context)
    return chain.stream(input={'question': query,
                                 'context': context})


