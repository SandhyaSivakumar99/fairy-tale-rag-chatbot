from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_embedding_store(docs):
    # 📌 Load embedding model
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 🧠 Create Chroma vector store
    vectordb = Chroma.from_documents(docs, embedding=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    return retriever
