from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_faiss_index(texts: list[str]):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_texts(texts, embeddings)

def retrieve_relevant_docs(vectorstore:FAISS, query:str, top_k:int=4):
    return vectorstore.similarity_search(query,k=top_k)
