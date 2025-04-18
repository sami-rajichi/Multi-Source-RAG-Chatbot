# 🧠 Multi-Source RAG Chatbot with Dynamic Retrieval

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multi-source-rag-chatbot.streamlit.app/)
[📺 Video Demo](https://youtu.be/TNGW4eHvOlE) | [🐛 Report Issues](https://github.com/sami-rajichi/Multi-Source-RAG-Chatbot/issues) | [📄 Full Documentation](https://github.com/sami-rajichi/Multi-Source-RAG-Chatbot/blob/main/Documentation___Multi_source_RAG_Chatbot_Project.pdf)

Welcome! This project showcases an intelligent **Retrieval-Augmented Generation (RAG)** chatbot designed to dynamically select the best information source to answer your questions. It intelligently decides whether to consult its internal AI knowledge base (vectorstore), perform a real-time web search, or rely on its foundational LLM capabilities, providing transparent and relevant responses.

## 🔍 Table of Contents

-   [✨ Key Features](#-key-features)
-   [📂 Project Structure](#-project-structure)
-   [🤖 How It Works: The Core Logic](#-how-it-works-the-core-logic)
-   [🚀 Try the Live Demo](#-try-the-live-demo)
-   [🛠️ Local Setup Guide](#%EF%B8%8F-local-setup-guide)
-   [🏃 Running the App Locally](#-running-the-app-locally)
-   [🧩 Technology Stack](#-technology-stack)
-   [🧠 Design Philosophy](#-design-philosophy)
-   [⚠️ Important Notes & Known Quirks](#%EF%B8%8F-important-notes--known-quirks)
-   [🔮 Future Roadmap](#-future-roadmap)
-   [🙏 Acknowledgments](#-acknowledgments)
-   [📜 License](#-license)

## ✨ Key Features

-   **🧠 Dynamic Routing:** Automatically determines the optimal retrieval strategy (Vectorstore, Web Search, or LLM Native) based on query analysis.
-   **🔍 Source Transparency:** Clearly indicates which retrieval mode was used for each response in the UI.
-   **⚙️ Manual Override:** Allows users to force a specific retrieval mode via the sidebar for targeted testing or specific needs.
-   **📄 Efficient Document Handling:** Ingests PDF and TXT files, performs intelligent chunking, and stores them efficiently in a vector database.
-   **🛡️ Robust Fallbacks:** Gracefully handles potential errors during retrieval (e.g., web search failure) by falling back to LLM Native mode when appropriate.
-   **📊 Integrated Evaluation:** Includes a Streamlit-based evaluation mode to benchmark performance against custom questions using LangSmith for detailed tracing.

## 📂 Project Structure

Here's a map of the key components:

```
Multi-Source-RAG-Chatbot/
├── data/                  # My source documents (PDFs/TXT) for the knowledge base. AI-focused data
├── src/                   # Main application source code.
│   ├── app.py             # The Streamlit user interface application.
│   ├── chatbot_graph.py   # Core LangGraph state machine defining the RAG logic and routing.
│   ├── data_loader.py     # Handles document loading, chunking, embedding, and vectorstore creation.
│   ├── tools.py           # Configuration for external tools (e.g., Tavily web search).
│   └── utils/             # Helper modules for env vars, files, chat UI, and evaluation.
├── tests/                 # Evaluation assets.
│   └── benchmark_questions.csv # Sample questions for the evaluation mode.
├── .env                   # Your local environment variables (API keys - **DO NOT COMMIT**).
├── run.sh                 # Convenience script for setup and execution.
├── Dockerfile             # Defines the Docker image.
├── docker-compose.yml     # Defines the Docker multi-container setup.
└── requirements.txt       # List of Python dependencies.
```

## 🤖 How It Works: The Core Logic

The chatbot's intelligence stems from the LangGraph state machine defined in `src/chatbot_graph.py`.

1.  **Initial Routing (`route_query_node`):**
    *   Analyzes the user query using an LLM call with a specialized prompt.
    *   Applies keyword overrides (e.g., "latest news" → Web Search, "Transformer architecture" → Vectorstore).
    *   Determines the initial best path: `vectorstore`, `web_search`, or `llm_native`.
    *   *If a mode is forced via the UI, this step is bypassed.*
2.  **Retrieval:**
    *   **Vectorstore:** Queries ChromaDB for relevant document chunks based on semantic similarity.
    *   **Web Search:** Uses the Tavily API to fetch relevant, up-to-date web snippets.
3.  **Generation (`generate_answer_rag_node` / `generate_llm_native_node`):**
    *   **RAG:** If documents/snippets were retrieved, an LLM generates the answer *strictly* based on the provided context.
    *   **LLM Native:** If no retrieval was needed or if retrieval failed (in automatic mode), the LLM generates the answer using its general knowledge and chat history.
4.  **State Management:** LangGraph manages the transitions between these steps, handles errors, and ensures the final state includes the answer and the mode used.

## 🚀 Try the Live Demo

Experience the chatbot instantly without any local setup:

-   [🌐 Live Streamlit Application](https://multi-source-rag-chatbot.streamlit.app/)
-   [🎬 Watch the Demo Video](https://youtu.be/TNGW4eHvOlE)

> **Note for Hosted App Users:** The hosted version uses shared resources.
>
> *   **Vectorstore:** You can rebuild or delete the vectorstore **once per session**. Changes are temporary and reset when the app instance restarts.
> *   **.env Configuration:** The hosted app allows uploading a temporary `.env` file for API keys. **For security, please use the "Delete .env file" button in the sidebar before closing your browser tab or navigating away.** Your uploaded file is temporary, but deleting it ensures it's promptly removed from the session.

## 🛠️ Local Setup Guide

### Prerequisites

-   Python `3.12.x` (tested with 3.12.9)
-   Git
-   API Keys:
    -   [Groq](https://console.groq.com/) (Free tier available)
    -   [Tavily Search API](https://tavily.com/) (Free research tier available)
    -   [LangSmith](https://smith.langchain.com/) (Optional, but highly recommended for tracing/evaluation)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sami-rajichi/Multi-Source-RAG-Chatbot.git
    cd Multi-Source-RAG-Chatbot
    ```

2.  **Set up environment variables:**
    *   Copy the example file:
        ```bash
        touch .env
        ```
    *   Edit the `.env` file using a text editor (like VS Code, Nano, Vim) and insert your API keys:
        ```dotenv
        # .env file contents
        GROQ_API_KEY="gsk_YOUR_GROQ_KEY"
        TAVILY_API_KEY="tvly-YOUR_TAVILY_KEY"

        # LangSmith (Optional - Needed for Evaluation Mode Tracing)
        LANGCHAIN_TRACING_V2="true"
        LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
        LANGCHAIN_API_KEY="ls__YOUR_LANGSMITH_KEY"
        LANGCHAIN_PROJECT="Multi-Source-RAG-Chatbot" # Or your preferred project name
        ```

3.  **(Optional) Add Documents:** Place your PDF/TXT files into the `data/` directory if you want to populate the vectorstore with custom knowledge. You will need to rebuild the vectorstore via the UI afterwards.

## 🏃 Running the App Locally

Choose your preferred method:

### 🚅 Method 1: Using the `run.sh` Script (Linux/macOS)

This script automates setup and execution. Ensure you have **Python 3.12+** and optionally **Docker** installed. It's highly recommended to run this in a clean directory or after cleaning up previous attempts.

```bash
chmod +x run.sh # Make executable (only needed once)
./run.sh [option]
```

**Available Options:**

-   **(No Option):** Performs full setup (dependency installation) and launches the Streamlit app. Assumes `.env` is configured.
-   `--install`: Installs dependencies only. Does not launch the app.
-   `--run`: Launches the Streamlit app. Assumes environment and dependencies are already set up.
-   `--docker`: Builds and runs the application using Docker Compose. Assumes Docker is running.
-   `--clean`: Clean up caches and temporary files.
-   `--help`: Displays the help message.

### 🏗️ Method 2: Manual Setup

For step-by-step control:

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv rag-env
    source rag-env/bin/activate  # Linux/Mac
    # or use: rag-env\Scripts\activate on Windows
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the Streamlit app:**
    ```bash
    streamlit run src/app.py
    ```

### 🐳 Method 3: Using Docker

Ensure Docker and Docker Compose are installed and running.

1.  **Build and run the containers:**
    ```bash
    docker-compose up --build -d # Use -d to run in detached mode
    ```
2.  Access the application in your browser at `http://localhost:8501`.
3.  To stop: `docker-compose down`

## 🧩 Technology Stack

| Component        | Technology             | Purpose & Rationale                                   |
| :--------------- | :--------------------- | :---------------------------------------------------- |
| **Core Logic**   | Langchain              | Foundation for RAG pipelines, component integration   |
|                  | LangGraph              | Manages complex, stateful agent/graph execution flows |
| **LLM**          | Groq API               | Provides access to fast LLMs (Llama 3, Mixtral)       |
| **Vector Store** | ChromaDB               | Local, efficient vector storage and similarity search |
| **Web Search**   | Tavily API             | Fetches relevant, concise real-time web results       |
| **UI**           | Streamlit              | Rapid development of interactive web application      |
| **Monitoring**   | LangSmith (Optional)   | Tracing, debugging, and evaluation of LLM apps      |
| **Container**    | Docker/Docker Compose  | Ensures consistent deployment environment             |

## 🧠 Design Philosophy

Our development was guided by these principles:

1.  **Intelligent Automation:** The core goal was dynamic routing – letting the system choose the best tool for the job without constant user intervention.
2.  **Transparency & Control:** While automation is key, users should understand *how* an answer was generated (source indication) and have the option to override the system (forced modes).
3.  **Modularity:** Separating concerns (UI, graph logic, data loading, utilities) makes the codebase easier to understand, maintain, and extend.
4.  **Resilience:** Incorporating error handling and fallbacks makes the application more robust against external service issues (e.g., API errors).

The `chatbot_graph.py` file embodies this philosophy, orchestrating the flow between different states (routing, retrieving, generating) based on context and logic.

## ⚠️ Important Notes & Known Quirks

-   **Windows File Locking:** Occasionally, on Windows, ChromaDB might encounter file locking issues if the database directory is accessed by another process.
    *   _Workarounds:_ Close other apps that might access the `chroma_db` directory, restart the Streamlit app, use WSL (Windows Subsystem for Linux), or restart the machine if persistent.
-   **Groq API Stability:** The free tier of Groq offers incredible speed but can sometimes experience `503 Internal Server Error` or rate limits during peak times.
    *   _Workarounds:_ Wait a minute and retry, check the [Groq Status Page](https://status.groq.com/), or consider their paid tiers for production use.
-   **Tavily API Limits:** The free research tier for Tavily has usage limits (e.g., searches per month). Monitor your usage if you rely heavily on web search.
-   **Vectorstore Persistence (Local):** The ChromaDB vectorstore created locally persists in the `chroma_db/` directory between runs. Use the UI options to manage it.
-   **Hosted App Limitations:** As mentioned in the [Try the Live Demo](#-try-the-live-demo) section, the public hosted app has temporary storage and requires mindful handling of uploaded `.env` files.

## 🔮 Future Roadmap

We're excited about potential future enhancements:

-   [ ] **Advanced RAG:** Implement techniques like HyDE or query rewriting for improved retrieval.
-   [ ] **Multi-Document QA:** Enable asking questions across multiple documents simultaneously.
-   [ ] **Agentic Tools:** Integrate more tools (calculator, code execution, specific APIs).
-   [ ] **UI Enhancements:** Add features like conversation saving/loading, user accounts.
-   [ ] **Streaming Responses:** Implement token streaming for faster perceived response times.

Feel free to suggest features or contribute!

## 🙏 Acknowledgments

This project wouldn't be possible without the amazing work of the open-source community and these key technologies:

-   The **Groq** team for democratizing access to high-speed LLM inference.
-   The **Langchain** & **LangGraph** developers for their powerful framework.
-   The **Streamlit** team for making web app development in Python so accessible.
-   The **ChromaDB** and **Tavily** teams for their excellent tools.
-   **You** for exploring this project!

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full details.

---

[⬆ Back to Top](#-table-of-contents)