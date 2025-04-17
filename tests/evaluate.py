from langchain_groq import ChatGroq
import pandas as pd
import time
import logging
import os
import uuid
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.schema.runnable.config import RunnableConfig
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Ensure config is loaded to set LangSmith vars if evaluate.py is run directly
try:
    # Assuming running from project root: python tests/evaluate.py
    from src.config import LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
    from src.chatbot_graph import build_chatbot_graph
except ImportError:
     # Assuming running from tests directory: python evaluate.py
     import sys
     sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add project root to path
     from src.config import LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
     from src.chatbot_graph import build_chatbot_graph


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - EVAL - %(levelname)s - %(message)s')

# --- Configuration ---
BENCHMARK_FILE = "tests/benchmark_questions.csv"
RESULTS_FILE = "tests/evaluation_results.csv"
SUMMARY_FILE = "tests/evaluation_summary.txt"
NUM_QUESTIONS = 3  # Target number of questions to evaluate
REQUEST_DELAY = 3    # Seconds between requests to avoid rate limiting


def print_summary(results_df: pd.DataFrame, summary_file: Optional[str] = None):
    """Generate and print evaluation summary statistics."""
    summary_lines = ["=== Evaluation Summary ==="]
    
    # Basic counts
    total_questions = len(results_df)
    summary_lines.append(f"Total Questions Processed: {total_questions}")
    
    # Error statistics
    error_count = results_df['error'].notna().sum()
    if error_count > 0:
        summary_lines.append(f"\nErrors Occurred: {error_count} ({error_count/total_questions:.1%})")
    
    # Mode accuracy
    if 'mode_correct' in results_df:
        mode_acc = results_df['mode_correct'].mean() * 100
        summary_lines.append(f"\nMode Accuracy: {mode_acc:.1f}%")
    
    # Latency statistics
    if 'latency_seconds' in results_df:
        avg_latency = results_df['latency_seconds'].mean()
        max_latency = results_df['latency_seconds'].max()
        min_latency = results_df['latency_seconds'].min()
        summary_lines.append(f"\nLatency Statistics:")
        summary_lines.append(f"- Average: {avg_latency:.2f}s")
        summary_lines.append(f"- Maximum: {max_latency:.2f}s")
        summary_lines.append(f"- Minimum: {min_latency:.2f}s")
    
    # Source usage statistics
    if 'sources_used' in results_df:
        sources_series = results_df['sources_used'].dropna()
        if not sources_series.empty:
            all_sources = []
            for sources in sources_series.str.split(', '):
                all_sources.extend(sources)
            source_counts = pd.Series(all_sources).value_counts()
            summary_lines.append(f"\nSource Usage Counts:")
            for source, count in source_counts.items():
                summary_lines.append(f"- {source}: {count}")
    
    # Format and output summary
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Save to file if specified
    if summary_file:
        try:
            with open(summary_file, 'w') as f:
                f.write(summary_text)
            logging.info(f"Evaluation summary saved to {summary_file}")
        except Exception as e:
            logging.error(f"Failed to save summary file: {e}")


def log_feedback_to_langsmith(
    client: Client,
    run_id: str,
    metrics: Dict[str, Any],
    question_id: Optional[int] = None
):
    """Helper function to log multiple feedback metrics to LangSmith."""
    if not client or not run_id:
        return
    
    try:
        # Log each metric in the dictionary
        for key, value in metrics.items():
            feedback_kwargs = {
                "run_id": run_id,
                "key": key,
            }
            
            # Handle different metric types appropriately
            if isinstance(value, bool):
                feedback_kwargs["score"] = 1 if value else 0
            elif isinstance(value, (int, float)):
                # Round numeric values to 4 decimal places for LangSmith compatibility
                feedback_kwargs["score"] = round(float(value), 4)
            else:
                feedback_kwargs["value"] = str(value)
            
            # Add question context if available
            if question_id is not None:
                feedback_kwargs["comment"] = f"Question ID: {question_id}"
            
            client.create_feedback(**feedback_kwargs)
        
        logging.info(f"Logged {len(metrics)} feedback metrics to LangSmith for run {run_id}")
    except Exception as fb_error:
        logging.error(f"Failed to log feedback to LangSmith: {fb_error}", exc_info=True)


def run_evaluation(chatbot_runnable = None):
    """Runs the chatbot against benchmark questions, logs results locally,
       and sends evaluation feedback to LangSmith."""
    logging.info("--- Starting Evaluation Run with LangSmith Feedback ---")
    
    # --- LangSmith Client Initialization ---
    client = None
    if LANGCHAIN_API_KEY:
        try:
            client = Client()
            logging.info(f"LangSmith client initialized. Project: '{LANGCHAIN_PROJECT}'")
        except Exception as e:
            logging.error(f"Failed to initialize LangSmith client: {e}", exc_info=True)
            logging.warning("Proceeding without programmatic LangSmith feedback.")
            client = None
    else:
        logging.warning("LANGCHAIN_API_KEY not found. LangSmith tracing and feedback disabled.")

    # --- Load Benchmark Data ---
    try:
        benchmark_df = pd.read_csv(BENCHMARK_FILE)
        # Fill NaN in optional columns for easier handling
        benchmark_df.fillna({
            'expected_mode': 'N/A',
            'question_id': ''
        }, inplace=True)
        
        # Ensure we have exactly NUM_QUESTIONS questions
        if len(benchmark_df) > NUM_QUESTIONS:
            benchmark_df = benchmark_df.head(NUM_QUESTIONS)
            logging.info(f"Truncated benchmark to first {NUM_QUESTIONS} questions")
        elif len(benchmark_df) < NUM_QUESTIONS:
            logging.warning(f"Only {len(benchmark_df)} questions found, expected {NUM_QUESTIONS}")
        
        logging.info(f"Loaded {len(benchmark_df)} questions from {BENCHMARK_FILE}")
    except FileNotFoundError:
        logging.error(f"Benchmark file not found at {BENCHMARK_FILE}. Evaluation cannot proceed.")
        return
    except Exception as e:
        logging.error(f"Error loading benchmark CSV: {e}", exc_info=True)
        return

    if 'question' not in benchmark_df.columns:
        logging.error("Benchmark CSV must contain a 'question' column.")
        return

    # --- Check Chatbot Availability ---
    if not chatbot_runnable:
        logging.error("Chatbot runnable is not available. Evaluation cannot proceed.")
        return

    # --- Evaluation Loop ---
    results = []
    total_questions = len(benchmark_df)

    for index, row in benchmark_df.iterrows():
        question_id = row.get('question_id', index + 1)
        question = row['question']
        expected_mode = row.get('expected_mode', 'N/A').strip().lower()

        logging.info(f"--- Processing Question {question_id}/{total_questions} ---")
        logging.info(f"Query: {question}")
        logging.info(f"Expected Mode: {expected_mode}")

        start_time = time.time()
        final_state = None
        error_occurred = None
        run_id = None
        sources_used = None

        # Set up LangSmith Tracing Callback
        ls_tracer = LangChainTracer(project_name=LANGCHAIN_PROJECT)
        run_config = RunnableConfig(callbacks=[ls_tracer])

        try:
            # Input for the graph
            inputs = {"query": question, "chat_history": []}

            # Invoke the graph with the tracer config
            final_state = chatbot_runnable.invoke(inputs, config=run_config)

            # Get Run ID
            if hasattr(ls_tracer, 'latest_run') and ls_tracer.latest_run:
                run_id_obj = ls_tracer.latest_run.id
                run_id = str(run_id_obj) if isinstance(run_id_obj, uuid.UUID) else run_id_obj
                logging.info(f"LangSmith Run ID captured: {run_id}")

            # Extract results from state
            answer = final_state.get("answer", "Error: No answer found in state.")
            actual_mode = final_state.get("retrieval_mode", "unknown").strip().lower()
            error_msg = final_state.get("error")

            # Extract source documents if available
            if "source_documents" in final_state:
                sources = []
                for doc in final_state["source_documents"]:
                    source = doc.metadata.get("source", "unknown")
                    if source not in sources:  # Avoid duplicates
                        sources.append(source)
                sources_used = ", ".join(sources) if sources else None
                logging.info(f"Sources used: {sources_used}")

            if error_msg:
                error_occurred = error_msg
                logging.error(f"Graph reported error: {error_msg}")
                actual_mode = "error"

        except Exception as e:
            logging.error(f"Exception occurred invoking chatbot: {e}", exc_info=True)
            answer = f"Error during invocation: {e}"
            actual_mode = "invocation_error"
            error_occurred = str(e)
            if hasattr(ls_tracer, 'latest_run') and ls_tracer.latest_run:
                run_id_obj = ls_tracer.latest_run.id
                run_id = str(run_id_obj) if isinstance(run_id_obj, uuid.UUID) else run_id_obj

        end_time = time.time()
        latency = end_time - start_time

        # --- Perform Evaluations ---
        # Mode Correctness
        mode_correct = None
        if expected_mode != 'n/a' and actual_mode not in ["error", "invocation_error"]:
            mode_correct = (expected_mode == actual_mode)
            logging.info(f"Mode Check: {'Correct' if mode_correct else 'Incorrect'} (Expected: {expected_mode}, Got: {actual_mode})")

        # --- Log Feedback to LangSmith ---
        if client and run_id:
            metrics = {
                "latency_seconds": latency,
                "sources_used": sources_used,
            }
            
            if mode_correct is not None:
                metrics["mode_correctness"] = mode_correct
                metrics["overall_correctness"] = mode_correct
            
            log_feedback_to_langsmith(client, run_id, metrics, question_id)

        # --- Store Local Results ---
        logging.info(f"Answer (Truncated): {answer[:150]}...")
        logging.info(f"Actual Mode: {actual_mode} | Latency: {latency:.2f}s")
        if error_occurred:
            logging.error(f"Error Detail: {error_occurred}")

        results.append({
            "question_id": question_id,
            "question": question,
            "expected_mode": expected_mode if expected_mode != 'n/a' else None,
            "actual_mode": actual_mode,
            "mode_correct": mode_correct,
            "answer": answer,
            "latency_seconds": latency,
            "sources_used": sources_used,
            "langsmith_run_id": run_id,
            "error": error_occurred
        })

        # Delay between requests
        time.sleep(REQUEST_DELAY)

    # --- Save Results and Generate Summary ---
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS_FILE, index=False, encoding='utf-8')
        logging.info(f"Evaluation results saved locally to {RESULTS_FILE}")
        
        # Generate and save summary
        print_summary(results_df, SUMMARY_FILE)
    except Exception as e:
        logging.error(f"Failed to save results: {e}", exc_info=True)

    logging.info("--- Evaluation Run Complete ---")
    print(f"\nEvaluation finished. Results saved to {RESULTS_FILE}")
    print(f"Summary saved to {SUMMARY_FILE}")
    if client:
        print(f"Check your LangSmith project '{LANGCHAIN_PROJECT}' for detailed traces and evaluation feedback.")


if __name__ == "__main__":
    print("Starting evaluation...")
    print("Ensure your .env file is configured, especially LangSmith variables.")
    print("Ensure the vector store exists (run 'python src/data_loader.py' if needed).")
    llm_instance = ChatGroq(
        temperature=0, 
        groq_api_key=os.getenv('GROQ_API_KEY'), 
        model_name=os.getenv('GROQ_MODEL_NAME')
    )
    chatbot_runable = build_chatbot_graph(llm_instance=llm_instance)
    run_evaluation(chatbot_runnable=chatbot_runable)