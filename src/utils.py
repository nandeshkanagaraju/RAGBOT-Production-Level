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
                # Set it in env so other LangChain components (like ChatOpenAI) can find it
                os.environ["OPENAI_API_KEY"] = api_key
        except (ImportError, FileNotFoundError):
            pass # Not running in Streamlit or no secrets file found

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Locally: Check your .env file. "
            "Streamlit Cloud: Add it to App Settings > Secrets."
        )
    
    return OpenAIEmbeddings(openai_api_key=api_key)
