import logging
try:
    from config import WEB_SEARCH_PROVIDER, TAVILY_API_KEY
except ImportError:
    from .config import DATA_PATH, CHROMA_PATH, EMBEDDING_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_web_search_tool(max_results=3):
    """Initializes and returns the configured web search tool."""
    logging.info(f"Initializing web search tool: {WEB_SEARCH_PROVIDER}")

    if WEB_SEARCH_PROVIDER == "tavily":
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY not found in environment variables.")
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            search_tool = TavilySearchResults(
                max_results=max_results, 
                tavily_api_key=TAVILY_API_KEY
            )
            logging.info(f"Tavily search tool initialized with max_results={max_results}.")
            return search_tool
        except ImportError:
            logging.error("Tavily library not installed. Please run: pip install tavily-python")
            raise
        except Exception as e:
            logging.error(f"Failed to initialize Tavily tool: {e}", exc_info=True)
            raise
    else:
        raise ValueError(f"Unsupported WEB_SEARCH_PROVIDER: {WEB_SEARCH_PROVIDER}. Choose 'tavily' or 'serper'.")

# if __name__ == "__main__":
#     try:
#         tool = get_web_search_tool()
#         print(f"Successfully initialized search tool: {type(tool)}")
#         # Test the tool
#         results = tool.invoke({"query": "Latest news on Groq LPUs"})
#         print("Test search results:", results)
#     except Exception as e:
#         print(f"Error initializing or testing search tool: {e}")