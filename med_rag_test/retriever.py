import os
import urllib

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from psycopg.generators import fetch

load_dotenv()
db_pass = os.getenv('POSTGRES_PASSWORD')
encoded_password = urllib.parse.quote(db_pass, safe='')

connection = f"postgresql+psycopg://admin:{encoded_password}@localhost:5433/meds_db"
collection_name = "meds"

embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
vector_store = PGVector(
    embeddings=embeddings,
    connection=connection,
    collection_name=collection_name,
    create_extension = False
)

def embed_docs():
    df = pd.read_csv('medicine_data.csv').dropna().tail(1000)
    print(len(df))
    documents=[
            Document(page_content=row['medicine_desc'], metadata=row.to_dict())
            for _, row in df.iterrows()
        ]
    vector_store.add_documents(documents)
    print("Embedding done")

def get_retriever(k=4):
    return vector_store.as_retriever(k=k)

if __name__ == '__main__':
    # resp = get_retriever().invoke('high bp')
    # print(resp)
    embed_docs()