import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LangSmith Configuration
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# Ensure LangSmith tracing is enabled if the key is set
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "meta-llama/llama-4-maverick-17b-128e-instruct")

# Web Search Configuration
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "tavily").lower()

# File Paths
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"

# Validate essential configurations
if not GROQ_API_KEY and os.path.exists('**/.env'):
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in the .env file.")

print("Config loaded:")
print(f" - GROQ Model: {GROQ_MODEL_NAME}")
print(f" - Embedding Model: {EMBEDDING_MODEL_NAME}")
print(f" - Web Search Provider: {WEB_SEARCH_PROVIDER}")
print(f" - LangSmith Project: {LANGCHAIN_PROJECT if LANGCHAIN_API_KEY else 'Not Configured'}")