import os
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class RAG:
    def __init__(self, model, retriever):
        self.retriever = retriever
        self.model = model
        self.chain = self._create_rag_chain()

    @classmethod
    def embed_docs(cls, file: bytes):
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file)
            loader = PyPDFLoader(temp_file.name)
            data = loader.load()
        os.unlink(temp_file.name)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=300)
        splits = splitter.split_documents(data)
        embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(splits)
        retriever = vector_store.as_retriever(k=4)
        print("Embedding Done")
        return retriever

    def _create_rag_chain(self):
        template = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Keep the answer concise.
        Question: {question} 
        Context: {context} 
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model | StrOutputParser()
        return chain

    def chat(self, query):
        context = self.retriever.invoke(query)
        response = self.chain.stream({"question": query, "context": context})
        return response
