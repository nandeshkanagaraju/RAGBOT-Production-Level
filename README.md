# 🤖 RAG Bot

A production-ready **Retrieval-Augmented Generation (RAG)** system built with Python, LangChain, and OpenAI. This application allows users to upload PDF documents and ask questions, receiving accurate answers based *only* on the provided context.

**[🚀 Live Demo](https://ragbot-application-nandesh.streamlit.app/)**

## ✨ Features
- **📄 PDF Ingestion**: Upload and process multiple PDF documents instantly.
- **🔍 Semantic Search**: Uses FAISS vector database to retrieve the most relevant context.
- **💡 Smart Suggestions**: Automatically generates follow-up questions based on your documents.
- **💬 Interactive Chat**: Streamlit-based interface for seamless Q&A.

## 🛠️ Tech Stack
- **LLM**: OpenAI GPT models
- **Framework**: LangChain, Streamlit
- **Vector DB**: FAISS
- **Processing**: PyPDF for text extraction

## 🚀 Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nandeshkanagaraju/RAGBOT-Production-Level.git
   cd RAGBOT-Production-Level
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   Create a `.env` file and add your OpenAI API Key:
   ```bash
   OPENAI_API_KEY=sk-your-key-here
   ```

4. **Run the App**
   ```bash
   streamlit run streamlit_app.py
   ```
