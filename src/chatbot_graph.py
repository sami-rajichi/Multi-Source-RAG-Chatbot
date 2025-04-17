import logging
import re
from typing import List, TypedDict, Annotated, Sequence, Optional, Dict, Any
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Import functions and config from other modules
try:
    from .config import WEB_SEARCH_PROVIDER
    from .data_loader import get_vectorstore_retriever
    from .tools import get_web_search_tool
except ImportError:
    from config import WEB_SEARCH_PROVIDER
    from data_loader import get_vectorstore_retriever
    from tools import get_web_search_tool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- State Definition ---
class ChatbotState(TypedDict):
    query: str
    documents: List[str]
    chat_history: Annotated[List[Dict[str, str]], operator.add]
    answer: str
    # Reflects the final mode determined or forced
    retrieval_mode: str  # 'llm_native', 'vectorstore', 'web_search', 'error'
    # User override from UI
    forced_mode: Optional[str]  # 'llm_native', 'vectorstore', 'web_search', or None
    # Tracks which retrieval was attempted for RAG node (set before generation)
    generation_source: Optional[str]  # 'vectorstore' or 'web_search'
    error: Optional[str]
    # REMOVED: docs_are_relevant

# --- LLM Initialization ---
llm = None  # Global variable updated by build_chatbot_graph

# --- Prompts ---

# 1. Comprehensive Router Prompt (for Automatic Mode)
COMPREHENSIVE_ROUTER_PROMPT_TEMPLATE = """You are an expert query routing assistant. Analyze the User Query and Chat History to determine the single best method to answer the query. Choose ONLY ONE: 'llm_native', 'vectorstore', or 'web_search'.

**Method Guidelines:**

1. **llm_native:**
    * Choose this ONLY for:
        * Purely conversational queries (greetings, thanks, chit-chat).
        * Simple common knowledge questions **unrelated** to AI-specific details (e.g. capitals, basic math, famous people).
        * Simple creative tasks (poems, jokes).
        * Inquiries about your own capabilities.
        * Simple clarifications of previous turns.
    * **Do NOT choose this if the query asks about ANY specific AI concept, model, technique, paper, or requires up-to-date information.**

2. **vectorstore:**
    * Choose this if the query asks for details, explanations, definitions, comparisons, or workings of AI concepts, models, techniques, algorithms, or papers that are covered by the internal AI document knowledge base.
    * Prioritize this for queries likely covered by the project’s AI documents. Keywords include: "Transformer", "RAG", "attention", "fine-tuning", "vector search", "Mixtral", "BERT", "explain", "how does", "paper", "model", "algorithm", "AI" "ML", "DL"...
    * **Do NOT choose** if the query requires real-time info or includes current events.

3. **web_search:**
    * Choose this if the query requires up-to-date information (news, prices, current events, release dates).
    * Choose this if the query is clearly outside of the internal AI domain or requires context beyond static documents.
    * Also select this if the query mentions keywords like "latest", "news", "current", or "today".

**Additional Examples:**
* "Hi there" -> llm_native
* "Capital of Canada?" -> llm_native
* "What is the Transformer architecture?" -> vectorstore
* "What is the ai" -> vectorstore
* "Explain the mechanism of attention in Transformer models." -> vectorstore
* "Latest news on OpenAI's GPT-4?" -> web_search
* "Compare Groq speed vs Nvidia H100." -> web_search
* "Tell me about the Mixtral paper in the docs." -> vectorstore
* "History of the internet?" -> web_search

Chat History: {chat_history}
User Query: {query}
Decision (llm_native, vectorstore, or web_search):
"""

COMPREHENSIVE_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", COMPREHENSIVE_ROUTER_PROMPT_TEMPLATE),
])

# RAG Prompts (Strict)
VECTORSTORE_RAG_PROMPT_TEMPLATE = """Answer based *only* on the provided context documents about AI... If you don't know, say so...
Context: {context} Question: {question} Answer:"""
VECTORSTORE_RAG_PROMPT = ChatPromptTemplate.from_template(VECTORSTORE_RAG_PROMPT_TEMPLATE)

WEB_SEARCH_RAG_PROMPT_TEMPLATE = """Answer based *only* on the provided web search results... If results are irrelevant, say so... Cite sources if possible...
Web Search Results: {context} Question: {question} Answer:"""
WEB_SEARCH_RAG_PROMPT = ChatPromptTemplate.from_template(WEB_SEARCH_RAG_PROMPT_TEMPLATE)

# --- Helper Functions ---
def format_chat_history_for_prompt(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return "No history yet."
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-6:]])

# --- Nodes ---

def route_query_node(state: ChatbotState) -> Dict[str, Any]:
    """Determines the single best method using the comprehensive router."""
    logger.info("--- Routing Query (Comprehensive LLM Router) ---")
    query = state['query']
    chat_history_str = format_chat_history_for_prompt(state['chat_history'])
    logger.debug(f"Routing query: '{query}'")
    try:
        router_chain = COMPREHENSIVE_ROUTER_PROMPT | llm
        response = router_chain.invoke({"query": query, "chat_history": chat_history_str})
        decision = response.content.strip().lower()
        logger.info(f"Initial comprehensive routing decision: {decision}")

        # --- Additional Override Logic Based on Query Keywords ---
        # Check for keywords indicating current events or up-to-date info
        if re.search(r'\b(latest|current|news|today|update)\b', query, re.IGNORECASE):
            logger.info("Overriding decision to 'web_search' based on query indicating current events or updates.")
            decision = "web_search"
        # If the query contains AI-specific keywords, override to vectorstore
        elif re.search(r'\b(transformer|rag|attention|fine[- ]?tuning|vector search|mixtral|bert|explain|how does|paper|model|algorithm)\b', query, re.IGNORECASE):
            logger.info("Overriding decision to 'vectorstore' based on AI topic keywords in query.")
            decision = "vectorstore"

        # Set retrieval_mode based on the final decision
        if "llm_native" in decision:
            final_decision = "llm_native"
        elif "vectorstore" in decision:
            final_decision = "vectorstore"
        elif "web_search" in decision:
            final_decision = "web_search"
        else:
            logger.warning(f"Comprehensive Router unexpected decision: '{decision}'. Defaulting to vectorstore.")
            final_decision = "vectorstore"  # Default preference

        logger.info(f"Final routing decision: {final_decision}")
        return {"retrieval_mode": final_decision}
    except Exception as e:
        logger.error(f"Error during comprehensive routing: {e}", exc_info=True)
        # Default to vectorstore even on error
        return {"retrieval_mode": "vectorstore", "error": f"Failed comprehensive routing, trying VS: {e}"}


def retrieve_vectorstore_node(state: ChatbotState) -> Dict[str, Any]:
    """Retrieves documents from vectorstore."""
    logger.info("--- Retrieving from Vectorstore ---")
    query = state['query']
    state['documents'] = []  # Clear previous docs
    try:
        retriever = get_vectorstore_retriever(k=7)  # Keep k=7 for better coverage
        documents = retriever.invoke(query)
        doc_contents = [doc.page_content for doc in documents]
        logger.info(f"Retrieved {len(doc_contents)} chunks from vectorstore.")
        if doc_contents:
            logger.debug(f"First retrieved doc snippet: {doc_contents[0][:200]}...")
        # Set source for RAG node, keep mode as vectorstore
        return {"documents": doc_contents, "generation_source": "vectorstore", "retrieval_mode": "vectorstore"}
    except Exception as e:
        logger.error(f"Error during vectorstore retrieval: {e}", exc_info=True)
        # Flag error, keep intended mode for edge logic
        return {"documents": [], "retrieval_mode": "vectorstore", "error": f"Vectorstore retrieval failed: {e}"}


def retrieve_web_node(state: ChatbotState) -> Dict[str, Any]:
    """Retrieves documents from web search."""
    logger.info("--- Retrieving from Web Search ---")
    query = state['query']
    current_mode = state['retrieval_mode']  # Should be 'web_search'
    state['documents'] = []  # Clear previous docs
    try:
        search_tool = get_web_search_tool(max_results=4)
        if WEB_SEARCH_PROVIDER == 'tavily':
            results = search_tool.invoke({"query": query})
            if isinstance(results, list) and results:
                results_str = "\n\n".join([f"Source URL: {res.get('url', 'N/A')}\nContent: {res.get('content', 'N/A')}" 
                                            for res in results if res.get('content')])
            elif isinstance(results, str):
                results_str = results
            else:
                results_str = ""
        else:
            raise NotImplementedError(f"Web provider {WEB_SEARCH_PROVIDER} not implemented.")

        retrieved_docs = [results_str] if results_str else []
        if retrieved_docs:
            logger.info(f"Retrieved web search results (length: {len(results_str)}).")
            # Set source and keep mode
            return {"documents": retrieved_docs, "generation_source": "web_search", "retrieval_mode": current_mode}
        else:
            logger.warning("Web search returned no results. Setting mode for fallback.")
            # Set mode explicitly to llm_native for the edge logic
            return {"documents": [], "retrieval_mode": "llm_native"}
    except Exception as e:
        logger.error(f"Error during web search retrieval: {e}", exc_info=True)
        # Set mode explicitly to llm_native for the edge logic on error
        return {"documents": [], "retrieval_mode": "llm_native", "error": f"Web search failed: {e}"}


def generate_answer_rag_node(state: ChatbotState) -> Dict[str, Any]:
    """Generates RAG answer using documents from specified source."""
    source = state.get('generation_source')
    if not source:
        # This indicates a logic error if reached without source being set
        error_msg = "RAG generation called without a valid source (vectorstore/web_search)."
        logger.error(error_msg)
        return {"answer": f"Internal Error: {error_msg}", "retrieval_mode": "error", "error": error_msg}

    mode = source  # Final mode is determined by the source used
    prompt_template = VECTORSTORE_RAG_PROMPT if mode == 'vectorstore' else WEB_SEARCH_RAG_PROMPT
    logger.info(f"--- Generating RAG Answer (Source: {source}) ---")
    query = state['query']
    documents = state['documents']
    chat_history_str = format_chat_history_for_prompt(state['chat_history'])

    if not documents:  # Should be caught by retrieval nodes setting mode to llm_native
        logger.error(f"generate_answer_rag_node called with empty documents (Source: {source}). Graph logic error.")
        return {"answer": "Internal Error: RAG generation called with no documents.", "retrieval_mode": "error", "error": f"RAG generation ({source}) called with no documents"}

    context = "\n\n".join(documents)
    logger.debug(f"Context length for LLM: {len(context)}")
    try:
        prompt = prompt_template.format(context=context, chat_history=chat_history_str, question=query)
        response = llm.invoke(prompt)
        answer = response.content
        # The RAG prompt instructs LLM to state if context is irrelevant.
        logger.info("LLM RAG generation complete.")
        # Set final mode based on source
        return {"answer": answer, "retrieval_mode": mode}
    except Exception as e:
        logger.error(f"Error during RAG answer generation (Source: {source}): {e}", exc_info=True)
        return {"answer": f"Sorry, error generating {source} RAG answer.", "retrieval_mode": "error", "error": f"{source.capitalize()} RAG generation failed: {e}"}


def generate_llm_native_node(state: ChatbotState) -> Dict[str, Any]:
    """Generates native LLM answer (initial route or fallback)."""
    logger.info(f"--- Generating Native LLM Answer ---")
    query = state['query']
    chat_history = state['chat_history']
    history_messages: List[BaseMessage] = []
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_messages.append(AIMessage(content=msg["content"]))

    native_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant..."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])
    try:
        chain = native_prompt | llm
        response = chain.invoke({"chat_history": history_messages, "query": query})
        answer = response.content
        logger.info("LLM native generation complete.")
        return {"answer": answer, "documents": [], "retrieval_mode": "llm_native"}  # Set final mode
    except Exception as e:
        logger.error(f"Error during native LLM generation: {e}", exc_info=True)
        return {"answer": "Sorry, error generating native answer.", "retrieval_mode": "error", "error": f"Native generation failed: {e}"}


def handle_error_node(state: ChatbotState) -> Dict[str, Any]:
    """Handles error states."""
    logger.error(f"--- Entering Error State ---")
    error_message = str(state.get("error", "An unknown error occurred."))
    logger.error(f"Error details: {error_message}")
    user_facing_error = f"Sorry, an error occurred: {error_message}"
    # Update state with error mode and message
    final_state = {**state, "answer": user_facing_error, "retrieval_mode": "error", "error": error_message}
    return final_state

# --- Conditional Edge Logic ---

def route_or_generate(state: ChatbotState) -> str:
    """Decides the first action based on forced mode or initial routing."""
    forced_mode = state.get("forced_mode")
    # --- Force Mode Check ---
    if forced_mode:
        logger.info(f"Routing decision: Forced to {forced_mode}")
        if forced_mode == "llm_native":
            return "generate_llm_native"
        if forced_mode == "vectorstore":
            return "retrieve_vectorstore"
        if forced_mode == "web_search":
            return "retrieve_web"
        state["error"] = f"Invalid forced_mode: {forced_mode}"
        return "handle_error"

    # --- Automatic Routing ---
    else:
        logger.info("Routing decision: Automatic mode, executing router.")
        return "route_query"


def decide_after_router(state: ChatbotState) -> str:
    """Routes after the comprehensive LLM router node (in automatic mode)."""
    mode = state.get("retrieval_mode")  # Set by route_query_node
    if state.get("error"):
        return "handle_error"

    if mode == "llm_native":
        return "generate_llm_native"
    if mode == "vectorstore":
        return "retrieve_vectorstore"
    if mode == "web_search":
        return "retrieve_web"

    logger.error(f"Invalid mode '{mode}' after comprehensive router.")
    state["error"] = f"Invalid mode after comprehensive router: {mode}"
    return "handle_error"


def decide_after_vectorstore_retrieval(state: ChatbotState) -> str:
    """Decides action after vectorstore retrieval (NO validation)."""
    logger.info("--- Deciding action after vectorstore retrieval ---")
    forced_mode = state.get("forced_mode")
    error_occurred = state.get("error") is not None
    documents_found = bool(state.get("documents"))

    # Handle retrieval errors first
    if error_occurred:
        logger.warning(f"Error during vectorstore retrieval: {state.get('error')}.")
        if forced_mode == "vectorstore":
            logger.error("Forced vectorstore failed retrieval. Ending in error.")
            return "handle_error"  # Error out if forced
        else:
            logger.info("Falling back to LLM Native due to vectorstore retrieval error.")
            return "generate_llm_native"  # Fallback

    # If no error, check if documents were found
    if documents_found:
        logger.info("Vectorstore retrieval successful. Proceeding to RAG generation.")
        state['generation_source'] = 'vectorstore'  # Set source for RAG node
        return "generate_answer_rag"
    else:
        logger.warning("Vectorstore retrieval returned no documents.")
        if forced_mode == "vectorstore":
            logger.error("Forced vectorstore found no documents. Ending in error.")
            state["error"] = "Forced vectorstore mode, but no relevant documents found."
            return "handle_error"
        else:
            logger.info("Falling back to LLM Native because no vectorstore documents found.")
            return "generate_llm_native"  # Fallback


def decide_after_web_retrieval(state: ChatbotState) -> str:
    """Decides after web search retrieval."""
    forced_mode = state.get("forced_mode")
    mode = state.get("retrieval_mode")
    generation_source = state.get("generation_source")
    error_msg = state.get("error")

    # Check for error during web retrieval first
    if error_msg and "Web search failed" in error_msg:
        logger.warning("Web search failed.")
        if forced_mode == "web_search":
            logger.error("Forced web_search failed retrieval. Ending in error.")
            return "handle_error"
        else:
            logger.info("Routing to: generate_llm_native (fallback after web search failure)")
            return "generate_llm_native"

    # If no error, check if docs were found
    if generation_source == "web_search":
        logger.info("Routing to: generate_answer_rag (from web)")
        return "generate_answer_rag"
    elif mode == "llm_native":
        logger.info("Routing to: generate_llm_native (fallback after no web results)")
        return "generate_llm_native"
    else:
        logger.error(f"Unexpected state after web retrieval: mode='{mode}', source='{generation_source}'.")
        return "handle_error"

# --- Graph Definition ---
def build_chatbot_graph(llm_instance):
    """Builds the graph with simplified routing and NO validation."""
    logger.info("Building the chatbot graph (Simplified Routing, No Validation)...")
    graph = StateGraph(ChatbotState)

    global llm
    if llm_instance:
        llm = llm_instance
    else:
        raise ValueError("LLM instance must be provided")

    # Add nodes
    graph.add_node("route_query", route_query_node)  # Node for automatic routing decision
    graph.add_node("retrieve_vectorstore", retrieve_vectorstore_node)
    graph.add_node("retrieve_web", retrieve_web_node)
    graph.add_node("generate_answer_rag", generate_answer_rag_node)
    graph.add_node("generate_llm_native", generate_llm_native_node)
    graph.add_node("handle_error", handle_error_node)

    # Define Edges
    # Entry point checks for forced mode
    graph.set_conditional_entry_point(
        route_or_generate,  # This function checks forced_mode
        {
            "route_query": "route_query",         # If automatic
            "retrieve_vectorstore": "retrieve_vectorstore",  # If forced VS
            "retrieve_web": "retrieve_web",       # If forced Web
            "generate_llm_native": "generate_llm_native",  # If forced Native
            "handle_error": "handle_error",       # If forced mode is invalid
        }
    )

    # After automatic router decides
    graph.add_conditional_edges(
        "route_query",
        decide_after_router,
        {
            "retrieve_vectorstore": "retrieve_vectorstore",
            "retrieve_web": "retrieve_web",
            "generate_llm_native": "generate_llm_native",
            "handle_error": "handle_error",
        }
    )

    # After vectorstore retrieval (automatic or forced)
    graph.add_conditional_edges(
        "retrieve_vectorstore",
        decide_after_vectorstore_retrieval,  # Logic handles forced vs auto fallback
        {
            "generate_answer_rag": "generate_answer_rag",
            "generate_llm_native": "generate_llm_native",  # Fallback only in automatic mode
            "handle_error": "handle_error",  # Error out if forced and failed
        }
    )

    # After web retrieval (automatic or forced)
    graph.add_conditional_edges(
        "retrieve_web",
        decide_after_web_retrieval,  # Logic handles forced vs auto fallback
        {
            "generate_answer_rag": "generate_answer_rag",
            "generate_llm_native": "generate_llm_native",  # Fallback only in automatic mode
            "handle_error": "handle_error",  # Error out if forced and failed
        }
    )

    # End after generation or error
    graph.add_edge("generate_answer_rag", END)
    graph.add_edge("generate_llm_native", END)
    graph.add_edge("handle_error", END)

    # Compile
    app = graph.compile()
    logger.info("Chatbot graph compiled successfully.")
    return app

# --- Get Runnable ---
chatbot_runnable = None  # App builds it
