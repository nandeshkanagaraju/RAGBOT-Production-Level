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
    documents = []
    errors = []
    
    if not os.path.exists(DATA_PATH):
        return [], []

    for filename in os.listdir(DATA_PATH):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, filename)
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                errors.append(f"{filename}: {e}")
                
    return documents, errors

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

    documents, errors = load_documents()
    if not documents and not errors:
        return "No documents found in data directory."

    status_msg = ""
    if documents:
        chunks = split_text(documents)
        save_to_faiss(chunks)
        status_msg += f"Successfully processed {len(documents)} documents into {len(chunks)} chunks."
    else:
        status_msg += "No valid documents processed."

    if errors:
        pdf_errors = "; ".join([e.split(":")[0] for e in errors]) # Just show filenames to keep it short
        status_msg += f"\nSkipped {len(errors)} files due to errors: {pdf_errors}"
        
    return status_msg

def main():
    print(ingest_documents())

if __name__ == "__main__":
    main()
