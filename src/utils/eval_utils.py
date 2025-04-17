import time
import uuid
import pandas as pd
import streamlit as st
import os
from typing import Optional
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.schema.runnable.config import RunnableConfig

REQUEST_DELAY = 3    # Seconds between requests to avoid rate limiting

def run_evaluation(chatbot_runnable, benchmark_df: pd.DataFrame, num_questions: int, selected_model: str) -> Optional[pd.DataFrame]:
    """Runs evaluation on a subset of benchmark questions."""
    results = []
    client = None
    
    if os.getenv('LANGCHAIN_API_KEY'):
        try:
            client = Client()
        except Exception as e:
            st.error(f"Failed to initialize LangSmith client: {e}")
            client = None
    
    total_questions = len(benchmark_df.head(num_questions))
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for index, row in benchmark_df.head(num_questions).iterrows():
        question_id = row.get('question_id', index + 1)
        question = row['question']
        expected_mode = row.get('expected_mode', 'N/A').strip().lower()
        
        status_text.text(f"Processing question {index+1}/{total_questions}: {question[:50]}...")
        progress_bar.progress((index + 1) / total_questions)
        
        start_time = time.time()
        final_state = None
        error_occurred = None
        run_id = None
        sources_used = None
        
        ls_tracer = LangChainTracer(project_name=os.getenv('LANGCHAIN_PROJECT'))
        run_config = RunnableConfig(callbacks=[ls_tracer])
        
        try:
            inputs = {"query": question, "chat_history": [], "model_name": selected_model}
            final_state = chatbot_runnable.invoke(inputs, config=run_config)
            
            answer = final_state.get("answer", "Error: No answer found in state.")
            actual_mode = final_state.get("retrieval_mode", "unknown").strip().lower()
            error_msg = final_state.get("error")
            
            if "source_documents" in final_state:
                sources = []
                for doc in final_state["source_documents"]:
                    source = doc.metadata.get("source", "unknown")
                    if source not in sources:
                        sources.append(source)
                sources_used = ", ".join(sources) if sources else None
            
            if error_msg:
                error_occurred = error_msg
                actual_mode = "error"
                
        except Exception as e:
            answer = f"Error during invocation: {e}"
            actual_mode = "invocation_error"
            error_occurred = str(e)
        
        end_time = time.time()
        latency = end_time - start_time
        
        mode_correct = None
        if expected_mode != 'n/a' and actual_mode not in ["error", "invocation_error"]:
            mode_correct = (expected_mode == actual_mode)
        
        if client and hasattr(ls_tracer, 'latest_run') and ls_tracer.latest_run:
            run_id = str(ls_tracer.latest_run.id)
            try:
                feedback_kwargs = {
                    "run_id": run_id,
                    "key": "latency_seconds",
                    "score": round(float(latency), 4)
                }
                client.create_feedback(**feedback_kwargs)
                
                if mode_correct is not None:
                    client.create_feedback(
                        run_id=run_id,
                        key="mode_correctness",
                        score=1 if mode_correct else 0
                    )
            except Exception as fb_error:
                st.error(f"Failed to log feedback to LangSmith: {fb_error}")
        
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
        
        time.sleep(REQUEST_DELAY)
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def display_evaluation_results(results_df: pd.DataFrame):
    """Displays evaluation results with metrics and visualizations."""
    if results_df is None or results_df.empty:
        st.warning("No evaluation results to display.")
        return
    
    st.subheader("Evaluation Summary")
    
    total_questions = len(results_df)
    successful_runs = len(results_df[~results_df['error'].notna()])
    error_rate = (total_questions - successful_runs) / total_questions * 100
    
    mode_evaluable = results_df[results_df['expected_mode'].notna()]
    mode_accuracy = mode_evaluable['mode_correct'].mean() * 100 if not mode_evaluable.empty else None
    
    avg_latency = results_df['latency_seconds'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Questions", total_questions)
    col2.metric("Error Rate", f"{error_rate:.1f}%")
    if mode_accuracy is not None:
        col3.metric("Mode Accuracy", f"{mode_accuracy:.1f}%")
    
    st.metric("Average Latency", f"{avg_latency:.2f} seconds")
    
    st.subheader("Performance Metrics")
    tab1, tab2 = st.tabs(["Latency Distribution", "Mode Accuracy"])
    
    with tab1:
        st.bar_chart(results_df['latency_seconds'])
    
    with tab2:
        if mode_accuracy is not None:
            st.bar_chart(mode_evaluable['mode_correct'].value_counts().rename({True: 'Correct', False: 'Incorrect'}))
    
    if 'sources_used' in results_df and results_df['sources_used'].notna().any():
        st.subheader("Source Usage Analysis")
        sources_series = results_df['sources_used'].dropna()
        if not sources_series.empty:
            all_sources = []
            for sources in sources_series.str.split(', '):
                all_sources.extend(sources)
            source_counts = pd.Series(all_sources).value_counts()
            st.bar_chart(source_counts)
    
    st.subheader("Detailed Results")
    st.dataframe(results_df)