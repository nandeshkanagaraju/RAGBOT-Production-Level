import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

def get_embeddings():
    """
    Initializes and returns the OpenAI embeddings model.
    Checks environment variables and Streamlit secrets for OPENAI_API_KEY.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Check Streamlit secrets if not in env
    if not api_key:
        try:
            import streamlit as st
            if "OPENAI_API_KEY" in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
        except (ImportError, FileNotFoundError):
            pass

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Locally: Check your .env file. "
            "Streamlit Cloud: Add it to App Settings > Secrets."
        )
    
    # AGGRESSIVE CLEANING
    # 1. Strip whitespace (newlines, spaces, U+2028, etc.)
    api_key = api_key.strip()
    # 2. Strip quotes (if user included them in the value)
    api_key = api_key.strip('"').strip("'")
    
    # Set/Update env var with the clean key for other LangChain components
    os.environ["OPENAI_API_KEY"] = api_key

    return OpenAIEmbeddings(openai_api_key=api_key)
