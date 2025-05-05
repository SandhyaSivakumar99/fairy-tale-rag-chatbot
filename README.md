# fairy-tale-rag-chatbot
An interactive chatbot built using LangChain and Retrieval-Augmented Generation (RAG) to answer questions from classic fairy tales using LLMs, embeddings, and a Gradio interface.

> **Done By**: Sandhya Sivakumar  
> **Tech Stack**: LangChain, Mistral LLM, HuggingFace, ChromaDB, Gradio

## 🌟 Overview

This project demonstrates a domain-specific conversational chatbot that allows users to interact with **popular fairy tales** using **natural language questions**.  
We used **Retrieval-Augmented Generation (RAG)** to combine **semantic search** with **LLM-based generation**, ensuring that answers are document-grounded and context-aware.

## 📚 Data Source

- 10 PDF documents of classic fairy tales (Cinderella, Rapunzel, The Little Mermaid, etc.)
- Included tables with characters, events, and dialogues
- Chunked and embedded using HuggingFace models

## 🛠 Technologies Used

- `LangChain` – pipeline construction
- `HuggingFaceEmbeddings` – semantic chunk embeddings
- `ChromaDB` – vector storage and retrieval
- `Mistral API` – language model response generation
- `Gradio` – interactive web UI
- `Python`, `Colab`, `Streamlit (optional)`

## 💡 Key Features

- 📄 Loads and chunks PDF fairy tales
- 🤖 Uses semantic similarity to find relevant passages
- 💬 Responds to user queries in a friendly tone
- 🧠 Remembers previous user questions using `ConversationBufferMemory`
- 🖼️ Gradio-based UI with sample prompts, retry, and exit buttons

## 🧪 Sample Interactions

| Prompt | Response |
|--------|----------|
| "Who helped Rapunzel escape?" | "The prince helped Rapunzel escape from the tower..." |
| "Hi" | "🧚‍♀️ Hello! Ask me anything about fairy tales!" |
| "Bye" | "Bye! Have a great day!" |

### 🔍 Screenshots

<p float="left">
  <img src="screenshots/welcome_ui.png" width="300"/>
  <img src="screenshots/retrieval_query.png" width="300"/>
  <img src="screenshots/multi_turn_context.png" width="300"/>
</p>

## 🚀 How It Works

1. **Load & Chunk PDFs** using `PyPDFDirectoryLoader + RecursiveCharacterTextSplitter`
2. **Embed** chunks using `HuggingFaceEmbeddings`
3. **Store** vectors in `ChromaDB`
4. **Construct RAG Chain** using `ConversationalRetrievalChain`
5. **Deploy UI** using `Gradio ChatInterface`

## 🔐 API Setup

You’ll need a valid **Mistral API Key** to run this code:

```bash
export MISTRAL_API_KEY="your-key-here"
