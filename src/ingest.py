import os
# Workaround for OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils import get_embeddings

DATA_PATH = "data"
FAISS_PATH = "faiss_index"

def load_documents():
    """
    Loads documents from the data directory.
    Configured for .pdf files.
    """
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_text(documents):
    """
    Splits documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_faiss(chunks):
    """
    Saves document chunks to FAISS index.
    """
    embeddings = get_embeddings()
    
    # Create a new DB from the documents.
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_PATH)
    print(f"Saved {len(chunks)} chunks to {FAISS_PATH}.")

def ingest_documents():
    """
    Ingests documents from the data directory and saves to FAISS.
    Returns a success message or raises an error.
    """
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return "Created data directory. Please upload files."

    documents = load_documents()
    if not documents:
        return "No documents found in data directory."

    chunks = split_text(documents)
    save_to_faiss(chunks)
    
    # Save chunks for BM25
    with open("chunks.pkl", "wb") as f:
        import pickle
        pickle.dump(chunks, f)
        
    return f"Successfully processed {len(documents)} documents into {len(chunks)} chunks."

def main():
    print(ingest_documents())

if __name__ == "__main__":
    main()
