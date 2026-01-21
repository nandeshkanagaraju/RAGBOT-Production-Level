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

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import pickle

def query_rag(query_text):
    """
    Queries the RAG system using Advanced techniques:
    1. Hybrid Search (BM25 + FAISS)
    2. Query Transformation (MultiQuery)
    3. Re-ranking (Contextual Compression)
    """
    embeddings = get_embeddings()
    llm = ChatOpenAI(temperature=0)
    
    # 1. Load FAISS (Vector)
    try:
        vector_db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    except Exception as e:
        return f"Error loading FAISS index: {e}\nDid you run `src/ingest.py` first?"

    # 2. Load BM25 (Keyword)
    try:
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 10
    except Exception:
        # Fallback to just FAISS if chunks.pkl missing
        print("Warning: chunks.pkl not found. Hybrid search disabled.")
        bm25_retriever = None

    # 3. Hybrid Search (Ensemble)
    if bm25_retriever:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6] # Weigh vector slightly higher
        )
        base_retriever = ensemble_retriever
    else:
        base_retriever = faiss_retriever

    # 4. Query Transformation (MultiQuery)
    # Generates variants of the question to improve recall
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    # 5. Re-ranking (Compression)
    # Filters the retrieved docs to keep only relevant parts
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever
    )

    # Execute Chain
    try:
        results = compression_retriever.invoke(query_text)
    except Exception as e:
        # Fallback if compression fails (e.g. empty results)
        print(f"Compression failed: {e}. Falling back to base retriever.")
        results = base_retriever.invoke(query_text)

    if not results:
        return "No matching results found."

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response = llm.invoke(prompt)
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
