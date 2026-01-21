import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

def get_embeddings():
    """
    Initializes and returns the OpenAI embeddings model.
    Ensure OPENAI_API_KEY is set in your environment or .env file.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    return OpenAIEmbeddings(openai_api_key=api_key)
