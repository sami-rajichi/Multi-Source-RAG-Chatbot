import os
import shutil
import logging
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import config variables
try:
    from config import DATA_PATH, CHROMA_PATH, EMBEDDING_MODEL_NAME
except ImportError:
    from .config import DATA_PATH, CHROMA_PATH, EMBEDDING_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_documents(data_path=DATA_PATH):
    """Loads documents from the specified directory using different loaders."""
    logging.info(f"Loading documents from {data_path}...")
    # Define loader types and their configurations
    loader_configs = [
        {"glob": "**/*.pdf", "loader_cls": PyPDFLoader, "loader_kwargs": {}},
        {"glob": "**/*.txt", "loader_cls": TextLoader, "loader_kwargs": {"encoding": "utf-8"}},
        # Add more loaders if needed (e.g., CSVLoader, JSONLoader)
    ]
    documents = []
    for config in loader_configs:
        try:
            loader = DirectoryLoader(
                data_path,
                glob=config["glob"],
                loader_cls=config["loader_cls"],
                loader_kwargs=config.get("loader_kwargs", {}),
                show_progress=True,
                use_multithreading=True # Can speed up loading
            )
            loaded_docs = loader.load()
            if loaded_docs:
                 logging.info(f"Loaded {len(loaded_docs)} documents using {config['loader_cls'].__name__}")
                 documents.extend(loaded_docs)
            else:
                 logging.info(f"No documents found for glob pattern {config['glob']}")

        except Exception as e:
            logging.error(f"Error loading files with {config['loader_cls'].__name__}: {e}", exc_info=True)

    if not documents:
        logging.warning("No documents were loaded. Ensure files exist in the data directory and loaders are configured correctly.")
    else:
         logging.info(f"Total documents loaded: {len(documents)}")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Splits documents into manageable chunks."""
    logging.info(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # Helpful for locating context
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(chunks)} chunks.")
    return chunks

def get_embedding_function():
    """Initializes and returns the embedding function."""
    logging.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    # Using HuggingFace embeddings (free, runs locally)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Or 'cuda' if GPU is available and configured
    )
    return embeddings

def setup_vectorstore(documents, chroma_path=CHROMA_PATH, embedding_function=None):
    """Creates and persists the ChromaDB vector store."""
    if not documents:
         logging.warning("No documents provided to setup_vectorstore. Skipping DB creation.")
         return None

    if embedding_function is None:
         embedding_function = get_embedding_function()

    # Check if the database directory already exists
    if os.path.exists(chroma_path):
        logging.warning(f"ChromaDB directory '{chroma_path}' already exists.")
        user_input = input("Do you want to delete the existing database and rebuild it? (y/N): ").strip().lower()
        if user_input == 'y':
            logging.info(f"Deleting existing ChromaDB at {chroma_path}...")
            shutil.rmtree(chroma_path)
            logging.info("Existing database deleted.")
        else:
            logging.info("Skipping vector store creation. Using existing database.")
            # Return the existing store
            try:
                vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
                logging.info("Successfully loaded existing ChromaDB.")
                return vectorstore
            except Exception as e:
                logging.error(f"Failed to load existing ChromaDB: {e}. Please check the directory or consider rebuilding.", exc_info=True)
                return None


    logging.info(f"Creating new ChromaDB vector store at {chroma_path}...")
    chunks = split_documents(documents)
    if not chunks:
        logging.error("No chunks generated from documents. Cannot create vector store.")
        return None

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=chroma_path
        )
        logging.info(f"Vector store created successfully and persisted at {chroma_path}.")
        return vectorstore
    except Exception as e:
        logging.error(f"Failed to create ChromaDB vector store: {e}", exc_info=True)
        return None

def get_vectorstore_retriever(k=4, chroma_path=CHROMA_PATH, embedding_function=None):
    """Loads the vector store and returns a retriever."""
    if not os.path.exists(chroma_path):
        logging.error(f"ChromaDB path '{chroma_path}' does not exist. Run setup_vectorstore first.")
        raise FileNotFoundError(f"ChromaDB not found at {chroma_path}")

    if embedding_function is None:
         embedding_function = get_embedding_function()

    try:
        vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': k, 'lambda_mult': 0.25})
        logging.info(f"ChromaDB retriever loaded successfully with k={k}.")
        return retriever
    except Exception as e:
        logging.error(f"Failed to load ChromaDB or create retriever: {e}", exc_info=True)
        raise

# --- Main Execution Block ---
# # Allows running this script directly to build the vector store
# if __name__ == "__main__":
#     print("--- Starting Vector Store Setup ---")
#     # 1. Load documents
#     docs = load_documents()
#     if docs:
#         # 2. Setup vector store (this will handle splitting and embedding)
#         setup_vectorstore(docs)
#         print("--- Vector Store Setup Complete ---")
#     else:
#         print("--- No documents found, skipping Vector Store Setup ---")