# 🧠 Multi-Source RAG Chatbot with Dynamic Retrieval

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-public-link.streamlit.app)  
[📺 Video Demo](https://youtu.be/your-demo-video) | [🐛 Report Issues](https://github.com/sami-rajichi/Multi-Source-RAG-Chatbot/issues)

Hey there! I'm thrilled to share this **intelligent RAG chatbot** I've been building. It's like having a research assistant that knows when to use its knowledge base, when to search the web, and when to rely on its own smarts. The best part? It makes these decisions automatically based on your questions!

## 🔍 Table of Contents
- [✨ Key Features](#-key-features)
- [📂 Project Structure](#-project-structure)
- [🤖 How It Works](#-how-it-works)
- [🚀 Try It Now](#-try-it-now)
- [🛠️ Setup Guide](#%EF%B8%8F-setup-guide)
- [🏃 Running the App](#-running-the-app)
- [🧩 Tech Stack](#-tech-stack)
- [🧠 Our Approach](#-our-approach)
- [⚠️ Known Quirks](#%EF%B8%8F-known-quirks)
- [🔮 What's Next?](#-whats-next)
- [🙏 Acknowledgments](#-acknowledgments)
- [📜 License](#-license)

## ✨ Key Features
Here's what makes this chatbot special:

- **Smart Routing** - Automatically chooses between three retrieval methods based on your query type
- **Transparent** - Shows exactly which method was used (documents, web search, or pure LLM)
- **Flexible** - Need specific results? Force a retrieval mode via the sidebar
- **Document Smart** - Handles PDFs and TXT files with clean chunking
- **Error-Resilient** - Graceful fallbacks when things don't go as planned

## 📂 Project Structure
Let me walk you through the important bits:

```
Multi-Source-RAG-Chatbot/
├── data/                  # Where your documents live (PDFs/TXT)
├── src/                   # The brains of the operation
│   ├── app.py             # Friendly Streamlit interface
│   ├── chatbot_graph.py   # The decision-making engine
│   ├── data_loader.py     # Document processor extraordinaire
│   ├── tools.py           # Web search toolkit
│   └── utils/             # Handy helpers
├── run.sh                # My one-click wonder script
├── Dockerfile            # For container fans
├── docker-compose.yml    # Docker orchestra
└── requirements.txt      # Python package list
```

## 🤖 How It Works
The magic happens in three stages:

1. **Query Analysis**  
   The system checks:  
   - Is this about AI concepts? → Vectorstore  
   - Does it need current info? → Web search  
   - Just chatting? → Pure LLM  

2. **Retrieval Execution**  
   - For documents: Uses ChromaDB with careful chunking  
   - For web: Leverages Tavily's clean search results  

3. **Response Generation**  
   - Crafts answers tuned to the retrieval method  
   - Always cites sources when available  

## 🚀 Try It Now
No setup needed to play with it:
- [Live Demo](https://your-public-link.streamlit.app)  
- [Watch the 2-min Demo](https://youtu.be/your-demo-video)

## 🛠️ Setup Guide

### What You'll Need
- Python 3.12+ (I recommend 3.12.9 for stability)
- API keys for:
  - [Groq](https://console.groq.com/) (free tier available)
  - [Tavily](https://tavily.com/) (research tier works great)
  - [LangSmith](https://smith.langchain.com/) (optional but helpful)

### Installation Steps
1. Clone the repo:
   ```
   git clone https://github.com/sami-rajichi/Multi-Source-RAG-Chatbot.git
   cd Multi-Source-RAG-Chatbot
   ```

2. Set up your environment:
   ```
   cp .env.example .env
   # Now edit .env with your favorite text editor
   ```

## 🏃 Running the App
Pick your favorite method:

### 🚅 The Express Route (run.sh)
Before using `run.sh`, I recommend you to:

1. Use a **Virtual Environment**.
2. Have **Python 3.12+** available on your machine.
3. Have **Docker** installed if you tend to experiment with `--docker` option.
```
./run.sh [option]
```
Options:  
- No option: Full setup + launch  
- `--install`: Just setup the environment  
- `--run`: Launch already-configured app  
- `--docker`: Containerized experience  
- `--clean`: Nuclear option (wipes venv)  
- `--help`: Show help message

### 🏗️ Manual Setup
For those who like control:
1. Create your virtual environment:
   ```
   python -m venv rag-env
   source rag-env/bin/activate  # Linux/Mac
   rag-env\Scripts\activate     # Windows
   ```

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

3. Launch:
   ```
   streamlit run src/app.py
   ```

### 🐳 Docker Fans
```
docker-compose up --build
```
Then visit `127.0.0.1:8501`

## 🧩 Tech Stack
Here's the toolkit that makes this possible:

| Component       | Technology               | Why We Chose It              |
|-----------------|--------------------------|------------------------------|
| LLM             | Groq                     | Blazing fast inference       |
| Vectorstore     | ChromaDB                 | Simple yet powerful          |
| Web Search      | Tavily API               | Research-focused results     |
| Framework       | LangChain + LangGraph    | Robust RAG pipelines         |
| Framework       | LangSmith                | Tracking and evaluation      |
| UI              | Streamlit                | Quick, beautiful interfaces  |

## 🧠 Our Approach
When building this, we prioritized:

1. **Intelligent Routing**  
   The chatbot doesn't just guess - it uses a sophisticated decision tree to choose retrieval methods

2. **Graceful Degradation**  
   If the first choice fails, it automatically tries alternatives

3. **Transparency**  
   You always know which method was used via clear UI indicators

The real magic happens in `chatbot_graph.py` - it's like air traffic control for information retrieval!

## ⚠️ Known Quirks
Heads up about a few things:

1. **Windows File Locking**  
   Sometimes Windows gets possessive about files. If vectorstore operations fail:
   - Try closing other programs
   - Use WSL if possible
   - Or just restart your machine
   - Or simply re-run the streamlit app

2. **Groq's Growing Pains**  
   Their free tier is amazing but can be flaky:
   - "Internal Server Error"? Wait 30 seconds
   - Consistent failures? Check [Groq Status](https://status.groq.com/)

3. **Web Search Limits**  
   Tavily's free tier gives you 100 searches/week:
   - Use them wisely!
   - Upgrade if you need more

## 🔮 What's Next?
Here's what's cooking in the lab:

- [ ] Multi-document Q&A - ask across all your files at once
- [ ] Screenshot understanding - extract text from images
- [ ] Voice interface - talk to your chatbot
- [ ] Shared sessions - collaborate in real-time

## 🙏 Acknowledgments
This project stands on the shoulders of giants:

- The **Groq** team for their insane inference speeds
- **LangChain/Langraph** for making RAG approachable
- **Streamlit** for turning Python into beautiful apps
- **You** for checking out this project!

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---
 
[⬆ Back to Top](#-table-of-contents)