import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils import get_embeddings

# Set environment variable for OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_PATH = "data"
FAISS_PATH = "faiss_index"

def load_documents():
    """
    Load .pdf documents from the data directory.
    
    Returns:
        tuple: (documents, errors)
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
    Split documents into chunks for embedding.
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
    Create and save FAISS vector index from document chunks.
    """
    embeddings = get_embeddings()
    
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_PATH)
    print(f"Saved {len(chunks)} chunks to {FAISS_PATH}.")

def ingest_documents():
    """
    Orchestrate the ingestion process: load, split, and save documents.
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
        pdf_errors = "; ".join([e.split(":")[0] for e in errors])
        status_msg += f"\nSkipped {len(errors)} files due to errors: {pdf_errors}"
        
    return status_msg

def main():
    print(ingest_documents())

if __name__ == "__main__":
    main()
