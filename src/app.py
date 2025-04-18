import streamlit as st
import time
import os
import logging
import gc
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import atexit

# Import utility functions
from utils.env_utils import check_env_vars, clear_env_vars, create_env_template
from utils.file_utils import force_delete_directory, delete_data_files, count_data_files, safe_filename
from utils.chat_utils import typewriter, get_chat_message_caption
from utils.eval_utils import run_evaluation, display_evaluation_results
from utils.custom_css import CSS_STYLE

# Import project components
try:
    from config import CHROMA_PATH, DATA_PATH
    from chatbot_graph import build_chatbot_graph
    from data_loader import setup_vectorstore, load_documents
except ImportError as e:
    st.error(f"Import Error: {e}. Check file paths and dependencies.")
    st.stop()

# Constants
AVAILABLE_GROQ_MODELS = [
    "deepseek-r1-distill-qwen-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-specdec",
    "llama-3.3-70b-versatile",
    "qwen-qwq-32b",
    "qwen-2.5-32b",
    "deepseek-r1-distill-llama-70b",
]

MODE_OPTIONS = ["Dynamic (Default)", "LLM Native Only", "Vectorstore Only", "Web Search Only"]
MODE_MAP = {
    "Dynamic (Default)": None,
    "LLM Native Only": "llm_native",
    "Vectorstore Only": "vectorstore",
    "Web Search Only": "web_search",
}

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
@st.cache_resource(show_spinner="Initializing Chatbot Engine...")
def get_chatbot_runnable(_llm_model_name: str):
    """Initialize and cache the chatbot runnable."""
    logger.info(f"Building graph for model: {_llm_model_name}")
    try:
        if not os.getenv('GROQ_API_KEY'): 
            st.error("GROQ_API_KEY not found.")
            return None
            
        llm_instance = ChatGroq(
            temperature=0, 
            groq_api_key=os.getenv('GROQ_API_KEY'), 
            model_name=_llm_model_name
        )
        return build_chatbot_graph(llm_instance=llm_instance)
    except Exception as e:
        logger.error(f"Failed to build graph: {e}", exc_info=True)
        st.error(f"Failed to initialize chatbot engine: {e}")
        return None

def cleanup():
    """Delete .env file when the app exits"""
    if os.path.exists('./.env'):
        try:
            os.remove('./.env')
            clear_env_vars()
            get_chatbot_runnable.clear()
            gc.collect()
        except Exception as e:
            st.error(f"Couldn't delete .env file: {e}")

def delete_vectorstore():
    """Handle vectorstore deletion with cleanup."""
    if not os.path.exists(CHROMA_PATH):
        st.warning("ℹ️ Vectorstore not found.")
        return
    
    try:
        # Clear caches and collect garbage
        get_chatbot_runnable.clear()
        gc.collect()
        
        if force_delete_directory(CHROMA_PATH):
            st.success("✅ Vectorstore deleted successfully!")
            if 'vectorstore_checked' in st.session_state: 
                del st.session_state.vectorstore_checked
        else:
            st.error("❌ Failed to delete vectorstore. Please restart the app and try again.")
    except Exception as e:
        st.error(f"❌ Error preparing to delete vectorstore: {e}")

def rebuild_vectorstore_with_detailed_status():
    """Rebuild vectorstore with detailed progress updates."""
    data_dir_exists = os.path.exists(DATA_PATH) and os.path.isdir(DATA_PATH)
    files_to_process = [entry for entry in os.scandir(DATA_PATH) if entry.is_file()] if data_dir_exists else []
    
    if not files_to_process: 
        st.error("Cannot rebuild: No data files found.")
        return
    
    with st.container():
        status_container = st.empty()
        with status_container.container():
            status = st.status("Rebuilding Vectorstore...", expanded=True)
            
            with status:
                total_files = len(files_to_process)
                try:
                    if os.path.exists(CHROMA_PATH):
                        status.write("⏳ Deleting existing vectorstore...")
                        if force_delete_directory(CHROMA_PATH):
                            if 'vectorstore_checked' in st.session_state: 
                                del st.session_state.vectorstore_checked
                            status.write("✅ Existing vectorstore deleted.")
                        else:
                            st.error("❌ Failed to delete vectorstore. Please restart the app and try again.")
                        time.sleep(0.5)

                    status.write(f"⏳ Loading {total_files} documents...")
                    progress_bar = st.progress(0.0)
                    
                    for idx, _ in enumerate(files_to_process):
                        time.sleep(0.02)
                        progress_bar.progress((idx + 1) / total_files)
                    
                    with st.spinner("Consolidating loaded documents..."): 
                        docs = load_documents()
                    
                    if not docs: 
                        status.update(label="Rebuild Failed!", state="error")
                        st.error("Failed to load docs.")
                        return
                    
                    status.write(f"✅ Loaded {len(docs)} document sections total.")
                    time.sleep(0.5)
                    
                    status.write("⏳ Indexing documents...")
                    with st.spinner("Indexing... please wait."): 
                        setup_vectorstore(docs)
                    
                    status.update(label="✅ Vectorstore Rebuilt Successfully!", state="complete", expanded=False)
                    
                    if 'vectorstore_checked' in st.session_state: 
                        del st.session_state.vectorstore_checked
                    get_chatbot_runnable.clear()
                    gc.collect()
                    time.sleep(2)
                    st.rerun()
                    
                except Exception as e:
                    status.update(label="Rebuild Failed!", state="error")
                    st.error(f"Error during rebuild: {e}")
                    logger.error(f"Error during rebuild: {e}", exc_info=True)

# --- Main App Code ---
def main():
    st.set_page_config(page_title="AI RAG Chatbot", page_icon="🧠", layout="centered")
    
    st.markdown(CSS_STYLE, unsafe_allow_html=True)
    
    # Initialize session state
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "chat"
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = os.getenv('GROQ_MODEL_NAME', AVAILABLE_GROQ_MODELS[2])
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Title and mode toggle
    st.title("🧠 Multi-Source AI Chatbot")
    st.caption(f"Using Model: {st.session_state.selected_model}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Chat Mode", use_container_width=True, disabled=st.session_state.app_mode == "chat"):
            st.session_state.app_mode = "chat"
            st.rerun()
    with col2:
        if st.button("📊 Evaluation Mode", use_container_width=True, disabled=st.session_state.app_mode == "evaluation"):
            st.session_state.app_mode = "evaluation"
            st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        config_tab, data_tab = st.tabs(["Chat Settings", "Data Management"])
        
        with config_tab:
            st.subheader("Environment Configuration")
            missing_vars = check_env_vars()
            
            if not missing_vars:
                st.markdown('<div class="env-var-status valid">✅ All required environment variables are properly configured</div>', unsafe_allow_html=True)
                if st.button(label='Delete .env file', type='tertiary'):
                    with st.empty().container():
                        st.markdown('<div class="env-var-status invalid">⚠️ Hold on a moment... Deletion in progress...</div>', unsafe_allow_html=True)
                        os.remove('./.env')
                        time.sleep(1)
                        clear_env_vars()
                        time.sleep(1)
                        get_chatbot_runnable.clear()
                        gc.collect()
                        time.sleep(2)
                        st.rerun()
            else:
                st.error(f'⚠️ Missing required variables: {", ".join(missing_vars)}')
            
            if missing_vars:
                env_template = create_env_template()
                st.download_button(
                    label="📥 Download .env Template",
                    data=env_template,
                    file_name=".env",
                    help="Download the template .env file to fill in your API keys"
                )
                
                uploaded_env = st.file_uploader("Upload your filled .env file")
                if uploaded_env is not None:
                    try:
                        # Ensure it really is a .env
                        if not uploaded_env.name.endswith(".env"):
                            time.sleep(3)
                            st.error("🚫 Invalid file: please upload a `.env` file.")
                        else:
                            try:
                                # Save to disk so load_dotenv can read it
                                with open("./.env", "wb") as f:
                                    f.write(uploaded_env.getvalue())

                                load_dotenv('./.env', override=True) 
                            except Exception as e:
                                st.error(f"Failed to load .env: {e}")
                        
                        new_missing = check_env_vars()
                        if not new_missing:
                            with st.empty().container():
                                st.success("✅ Valid .env file detected!")
                                time.sleep(1)
                                
                                # Clear all caches and reset app state
                                get_chatbot_runnable.clear()
                                gc.collect()
                                st.success("🔄 Reloading application with new configuration...")
                                time.sleep(1)
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error loading .env file: {e}")
            
            st.subheader("Model Settings")
            selected_model = st.selectbox(
                "LLM Model:", 
                options=AVAILABLE_GROQ_MODELS, 
                index=AVAILABLE_GROQ_MODELS.index(st.session_state.selected_model),
                key="model_select"
            )
            if st.session_state.selected_model != selected_model:
                st.session_state.selected_model = selected_model
                get_chatbot_runnable.clear()
            
            selected_mode_option = st.radio(
                "Retrieval Mode:", 
                options=MODE_OPTIONS, 
                index=0,
                key="selected_mode_option"
            )
            st.session_state.forced_mode = MODE_MAP[selected_mode_option]
        
        with data_tab:
            st.subheader("Vectorstore")
            vs_exists = os.path.exists(CHROMA_PATH)
            data_files_present = count_data_files(DATA_PATH) > 0
            
            col1, col2 = st.columns(2)
            col1.metric("Status", "Found" if vs_exists else "Not Found")
            col2.metric("Data Files", "Present" if data_files_present else "None")
            
            st.write("### Manage Vectorstore")
            if st.button("⚠️ Delete Vectorstore", disabled=not vs_exists):
                delete_vectorstore()
                st.rerun()
            
            if st.button("🔄 Rebuild Vectorstore", disabled=not data_files_present):
                rebuild_vectorstore_with_detailed_status()
                st.rerun()
            
            st.write("### Manage Data Files")
            if st.button("🗑️ Delete All Data Files", disabled=not data_files_present):
                deleted_count, failed_files = delete_data_files(DATA_PATH)
                if deleted_count > 0:
                    st.success(f"Deleted {deleted_count} file(s).")
                if failed_files:
                    st.warning(f"Could not delete: {', '.join(failed_files)}")
                st.rerun()
            
            st.write("### Add New Data")
            uploaded_files = st.file_uploader(
                "Upload PDF/TXT files", 
                type=["pdf", "txt"], 
                accept_multiple_files=True
            )
            if uploaded_files and st.button("Process Uploads & Rebuild VS"):
                if not os.path.exists(DATA_PATH):
                    os.makedirs(DATA_PATH)
                
                saved_files = 0
                for uploaded_file in uploaded_files:
                    safe_name = safe_filename(uploaded_file.name)
                    file_path = os.path.join(DATA_PATH, safe_name)
                    try:
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_files += 1
                    except Exception as e:
                        st.error(f"Error saving {safe_name}: {e}")
                
                if saved_files > 0:
                    st.success(f"Saved {saved_files} files.")
                    rebuild_vectorstore_with_detailed_status()

    # Initialize chatbot
    chatbot_runnable = get_chatbot_runnable(st.session_state.selected_model)
    if not chatbot_runnable:
        st.error("Chatbot engine failed to initialize.")
        st.stop()

    # Check vectorstore existence
    if 'vectorstore_checked' not in st.session_state:
        if not os.path.exists(CHROMA_PATH):
            st.warning(f"Vectorstore not found. Add data via sidebar.", icon="⚠️")
        st.session_state.vectorstore_checked = True

    # App mode content
    if st.session_state.app_mode == "chat":
        # Chat history display
        chat_history_container = st.container()
        with chat_history_container:
            st.markdown('<div class="chat-history-container">', unsafe_allow_html=True)
            
            for i, message in enumerate(st.session_state.messages):
                avatar = "👤" if message["role"] == "user" else "🤖"
                with st.chat_message(message["role"], avatar=avatar):
                    if (message["role"] == "assistant" and 
                        i == len(st.session_state.messages) - 1 and 
                        not message.get("already_typed", False)):
                        typewriter(st, message["content"])
                        st.session_state.messages[i]["already_typed"] = True
                    else:
                        st.markdown(message["content"])
                    
                    caption = get_chat_message_caption(message)
                    if caption:
                        st.caption(caption)
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Assistant response generation
        if (st.session_state.messages and 
            st.session_state.messages[-1]["role"] == "user" and 
            not st.session_state.get("assistant_processing", False)):
            
            st.session_state.assistant_processing = True
            
            with chat_history_container:
                with st.chat_message("assistant", avatar="🤖"):
                    thinking_placeholder = st.empty()
                    thinking_placeholder.markdown("Thinking... 🤔")
            
            user_prompt = st.session_state.messages[-1]["content"]
            final_mode = "unknown"
            error_occurred = False
            response_content = ""
            start_time = time.time()
            
            try:
                final_state = chatbot_runnable.invoke({
                    "query": user_prompt,
                    "chat_history": st.session_state.messages[:-1],
                    "forced_mode": st.session_state.get("forced_mode")
                })
                response_content = final_state.get("answer", "Sorry, I couldn't generate a response.")
                final_mode = final_state.get("retrieval_mode", "unknown")
                if final_state.get("error"):
                    final_mode = "error"
                    error_occurred = True
            except Exception as e:
                error_occurred = True
                final_mode = "invocation_error"
                response_content = f"An unexpected error occurred: {e}"

            processing_time = f"{(time.time() - start_time):.2f}s"
            thinking_placeholder.empty()

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_content,
                "mode": final_mode,
                "time": processing_time,
                "already_typed": False
            })

            st.session_state.assistant_processing = False
            st.rerun()

        # Chat input
        st.markdown('<div class="fixed-chat-container">', unsafe_allow_html=True)
        with st.container():
            cols = st.columns([9, 1])
            with cols[0]:
                prompt = st.chat_input("Ask me anything...", key="chat_input")
            with cols[1]:
                st.button("🗑️", key="clear_chat", on_click=lambda: st.session_state.update(messages=[]), help="Clear chat history")
        st.markdown('</div>', unsafe_allow_html=True)

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            if "assistant_processing" in st.session_state:
                del st.session_state.assistant_processing
            st.rerun()

    else:  # Evaluation Mode
        st.subheader("📊 Chatbot Evaluation")
        benchmark_file = st.file_uploader("Upload Benchmark CSV", type=["csv"])
        
        if benchmark_file:
            try:
                benchmark_df = pd.read_csv(benchmark_file)
                benchmark_df.fillna({'expected_mode': 'N/A'}, inplace=True)
                
                st.success(f"Loaded {len(benchmark_df)} benchmark questions")
                
                with st.expander("View Benchmark Data Sample"):
                    st.dataframe(benchmark_df)
                
                max_questions = len(benchmark_df)
                num_questions = st.slider(
                    "Number of questions to evaluate",
                    min_value=1,
                    max_value=max_questions,
                    value=min(25, max_questions)
                )
                
                if st.button("🚀 Start Evaluation", type="primary"):
                    with st.spinner("Running evaluation..."):
                        results_df = run_evaluation(
                            chatbot_runnable,
                            benchmark_df,
                            num_questions,
                            st.session_state.selected_model
                        )
                    
                    if results_df is not None:
                        st.session_state.evaluation_results = results_df
                        display_evaluation_results(results_df)
                        
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Evaluation Results",
                            data=csv,
                            file_name="evaluation_results.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Error loading benchmark file: {e}")
        
        elif 'evaluation_results' in st.session_state:
            st.info("Previous evaluation results loaded.")
            display_evaluation_results(st.session_state.evaluation_results)
            
            csv = st.session_state.evaluation_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Evaluation Results",
                data=csv,
                file_name="evaluation_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    # Register cleanup function
    atexit.register(cleanup)
    
    try:
        main()
    except Exception as e:
        st.error(f'Problem occurred. App stopped. {e}')
        # Force cleanup even on error
        if os.path.exists('.env'):
            try:
                os.remove('.env')
            except Exception as cleanup_error:
                st.error(f"Cleanup failed: {cleanup_error}")
        st.stop()