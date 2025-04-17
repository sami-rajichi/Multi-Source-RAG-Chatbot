import os
from dotenv import load_dotenv

# Required environment variables
REQUIRED_ENV_VARS = {
    "GROQ_API_KEY": "Your Groq API key",
    "TAVILY_API_KEY": "Your Tavily API key",
    "LANGCHAIN_API_KEY": "Your LangSmith API key",
    "LANGCHAIN_PROJECT": "Your LangSmith project name",
    "LANGCHAIN_TRACING_V2": "true/false",
    "LANGCHAIN_ENDPOINT": "LangSmith endpoint URL",
    "EMBEDDING_MODEL_NAME": "Default: sentence-transformers/all-MiniLM-L6-v2",
    "GROQ_MODEL_NAME": "Default: meta-llama/llama-4-maverick-17b-128e-instruct",
    "WEB_SEARCH_PROVIDER": "Default: tavily"
}

def check_env_vars():
    """Check if required environment variables are set."""
    load_dotenv()  # Reload environment variables
    missing = []
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            missing.append(var)
    return missing

def clear_env_vars():
    """Clear required environment variables"""
    for var in REQUIRED_ENV_VARS:
        if var in os.environ:
            os.environ.pop(var, None)

def create_env_template():
    """Create a template .env file content."""
    lines = []
    for var, description in REQUIRED_ENV_VARS.items():
        lines.append(f"# {description}")
        if var in ["EMBEDDING_MODEL_NAME", "GROQ_MODEL_NAME", "WEB_SEARCH_PROVIDER"]:
            lines.append(f"{var}={os.getenv(var, REQUIRED_ENV_VARS[var].split(': ')[1])}")
        else:
            lines.append(f"{var}=")
        lines.append("")
    return "\n".join(lines)