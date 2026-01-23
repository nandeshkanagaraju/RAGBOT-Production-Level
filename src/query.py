import sys
import os
# Workaround for OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils import get_embeddings

FAISS_PATH = "faiss_index"

PROMPT_TEMPLATE = """
Answer the question based only on the following context.
Provide a comprehensive and detailed answer, covering all relevant aspects found in the context.
If the context contains multiple related details, ensuring you explain the "how" and "why" if available.

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text):
    """
    Queries the RAG system and returns the answer.
    """
    # Prepare the DB.
    embeddings = get_embeddings()
    try:
        db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return f"Error loading FAISS index: {e}\nDid you run `src/ingest.py` first?"
    
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    
    if len(results) == 0:
        return "No matching results found."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI()
    response = model.invoke(prompt)
    
    return response.content

def generate_suggestions():
    """
    Generates 3 suggested questions based on the document context.
    """
    embeddings = get_embeddings()
    try:
        db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return []

    # Get a few random/top chunks to inspire questions
    # querying for "summary" or "main topics" usually gets diverse chunks
    results = db.similarity_search("main topics summary important facts", k=5)
    if not results:
        return []

    context_text = "\n\n".join([doc.page_content for doc in results])
    
    prompt = f"""
    Based on the following context, generate 3 interesting questions that a user might ask.
    Return ONLY the questions, separated by specific newline characters. Do not number them.
    
    Context:
    {context_text}
    """
    
    model = ChatOpenAI()
    response = model.invoke(prompt)
    
    questions = [q.strip() for q in response.content.split('\n') if q.strip()]
    return questions[:3]

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/query.py \"<question>\"")
        return

    query_text = sys.argv[1]
    response = query_rag(query_text)
    print(f"--- Response ---\n{response}\n")

if __name__ == "__main__":
    main()
