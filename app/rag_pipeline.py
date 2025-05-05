import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_mistralai.chat_models import ChatMistralAI

# 🔐 Set your API Key (replace with os.environ for safety)
os.environ["MISTRAL_API_KEY"] = "your-api-key-here"

# 🧠 Initialize the Mistral LLM
mistral_llm = ChatMistralAI(model="mistral-small", temperature=0)

# 📚 Load PDFs from 'data/' folder
loader = PyPDFDirectoryLoader("data")
documents = loader.load()

# ✂️ Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 🔎 Create Embeddings and Vector DB
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(docs, embedding=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# 🧵 Enable Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 🔗 Create Conversational Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=mistral_llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)
