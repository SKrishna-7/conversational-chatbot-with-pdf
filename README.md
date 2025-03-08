# Conversational RAG with PDF Uploads & Chat History

## 📌 Project Overview

**This project is a Conversational RAG (Retrieval-Augmented Generation) chatbot that allows users to upload PDF documents, extract relevant information, and engage in a dynamic conversation with historical context. The chatbot utilizes a retrieval-based approach to fetch relevant document segments before generating responses, making it highly effective for question-answering over custom documents.**

## 🚀 Features

- 📂 Upload multiple PDF documents
- 🔍 Extracts and indexes document content for efficient retrieval
- 💬 Conversational memory to maintain chat history
- 🤖 Uses a powerful LLM (Gemma-2-9b-It via Groq API) for response generation
- ⚡ Efficient vector search with ChromaDB
- 🏗 Streamlit-based interactive UI
  

## 🛠️ Tech Stack

- Python 🐍
- LangChain 🔗 (for retrieval-augmented generation)
- ChromaDB 📚 (for vector storage & retrieval)
- Streamlit 🎨 (for building the UI)
- Hugging Face Embeddings 🧠 (for text embedding)
- Groq API ⚡ (for running the LLM)
- PyPDFLoader 📄 (for processing PDFs)
