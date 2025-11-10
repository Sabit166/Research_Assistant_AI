# 🧠 Research Assistant AI with Memory

<div align="center">

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-green)
![Ollama](https://img.shields.io/badge/Ollama-qwen2.5:7b-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4%2B-purple)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**An intelligent research assistant that learns from your documents and remembers past conversations.**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📋 Overview

Research Assistant AI is a sophisticated document analysis and question-answering system built with **LangGraph**, **Ollama**, and **ChromaDB**. It features a dual-memory architecture that combines short-term session memory with persistent long-term knowledge storage, enabling contextual conversations about your documents that improve over time.

### 🎯 What Makes It Special?

- **Dual Memory System**: Short-term session cache + long-term vector database
- **PDF Intelligence**: Extract, chunk, and semantically understand PDF documents
- **Conversational**: Natural dialogue with context awareness
- **Local & Private**: Runs entirely on your machine with Ollama
- **Flexible Interface**: Web UI (Streamlit) or CLI for development
- **Production-Ready**: Comprehensive logging, error handling, and configuration

---

## ✨ Features

### Core Capabilities

- 📄 **PDF Processing**: Upload and parse PDF documents with intelligent text extraction
- 🔍 **Semantic Search**: Find relevant information using embedding-based similarity
- 💬 **Natural Conversations**: Ask follow-up questions with full context awareness
- 🧠 **Persistent Memory**: Automatically saves important sessions to long-term storage
- 📚 **Source Citations**: Every answer includes relevant sources with relevance scores
- 🔄 **Session Management**: Resume conversations and build on previous knowledge

### Technical Features

- **12-Node LangGraph State Machine**: Sophisticated workflow with conditional routing
- **ChromaDB Vector Database**: Fast, local vector storage for embeddings
- **Ollama Integration**: Use powerful local LLMs (qwen2.5:7b, llama3.2, etc.)
- **Sentence Transformers**: High-quality embeddings (all-MiniLM-L6-v2)
- **Pydantic Configuration**: Type-safe, validated settings
- **Comprehensive Logging**: Debug-level insights for development

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** ([Download](https://ollama.ai/download))
- **4GB+ RAM** (for model)
- **500MB+ Disk** (per 100-page PDF processed)

### Installation

1. **Clone the repository**
   ```powershell
   git clone https://github.com/Sabit166/Research_Assistant_AI.git
   cd Research_Assistant_AI
   ```

2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Install Ollama and download the model**
   ```powershell
   # Install Ollama from https://ollama.ai/download
   
   # Pull the LLM model
   ollama pull qwen2.5:7b
   ```

4. **Start Ollama server**
   ```powershell
   ollama serve
   ```
   Keep this terminal open!

5. **Launch the application**

   **Option A: Web UI (Recommended for end users)**
   ```powershell
   streamlit run app.py
   ```
   Opens in browser at `http://localhost:8501`

   **Option B: CLI (Recommended for developers)**
   ```powershell
   python cli.py
   ```
   Interactive command-line interface with debug capabilities

### First Steps

1. **Upload a PDF**: Use the sidebar (Web UI) or `/upload` command (CLI)
2. **Ask a question**: Type naturally - "What is this document about?"
3. **View sources**: See which parts of the document were used
4. **Continue conversation**: Ask follow-ups - "Tell me more about..."

📖 **Detailed guide**: See [QUICKSTART.md](QUICKSTART.md) for step-by-step tutorials

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Assistant AI                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │   Web UI     │         │     CLI      │                  │
│  │  (Streamlit) │         │  (Debug)     │                  │
│  └──────┬───────┘         └──────┬───────┘                  │
│         │                        │                           │
│         └────────────┬───────────┘                           │
│                      ↓                                       │
│         ┌────────────────────────┐                           │
│         │   LangGraph Engine     │                           │
│         │   (12-Node Workflow)   │                           │
│         └────────────────────────┘                           │
│                      ↓                                       │
│    ┌─────────────────┼─────────────────┐                    │
│    ↓                 ↓                 ↓                     │
│ ┌──────┐      ┌──────────┐      ┌──────────┐               │
│ │ PDF  │      │  Ollama  │      │ ChromaDB │               │
│ │Utils │      │   LLM    │      │  Vector  │               │
│ └──────┘      └──────────┘      └──────────┘               │
│    ↓               ↓                   ↓                     │
│ ┌──────────────────────────────────────────┐                │
│ │         Dual Memory System               │                │
│ │  ┌──────────────┐  ┌──────────────────┐  │                │
│ │  │ Short-Term   │  │   Long-Term      │  │                │
│ │  │ (Session)    │  │   (Persistent)   │  │                │
│ │  │ - Messages   │  │ - Vector DB      │  │                │
│ │  │ - Docs       │  │ - Summaries      │  │                │
│ │  └──────────────┘  └──────────────────┘  │                │
│ └──────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### LangGraph Workflow (12 Nodes)

<details>
<summary><b>Click to expand workflow diagram</b></summary>

```
START → session_init → check_input → [PDF or Query?]
                                           ↓           ↓
                                      ingest_pdf   process_query
                                           ↓           ↓
                                   confirm_ingest  retrieve_memory
                                           ↓           ↓
                                      merge_context ←─┘
                                           ↓
                                    generate_answer
                                           ↓
                                    update_session
                                           ↓
                                    should_persist? → [Yes/No]
                                           ↓              ↓
                                  persist_to_longterm    │
                                           ↓              ↓
                                      finalize_response ←┘
                                           ↓
                                          END
```

#### Node Descriptions

1. **session_init**: Initialize session ID and empty memory structures
2. **check_input**: Determine if input is PDF upload or query
3. **ingest_pdf**: Parse PDF, extract text, chunk, and embed
4. **confirm_ingest**: Confirm successful PDF processing
5. **process_query**: Parse and validate user query
6. **retrieve_memory**: Search both short-term and long-term memory
7. **merge_context**: Combine, deduplicate, and rank retrieved chunks
8. **generate_answer**: Use LLM to generate response from context
9. **update_session**: Add query/answer to session history
10. **should_persist**: Check if session should be saved (3+ interactions)
11. **persist_to_longterm**: Save session summary to vector database
12. **finalize_response**: Format final output with sources

</details>

### Dual Memory System

| Memory Type | Storage | Lifespan | Use Case |
|------------|---------|----------|----------|
| **Short-Term** | JSON files (`./data/sessions/`) | Current session | Recent messages, uploaded PDFs |
| **Long-Term** | ChromaDB (`./data/chromadb/`) | Permanent | Past conversations, document knowledge |

**Memory Flow**:
1. User uploads PDF → Stored in **both** memories
2. User asks questions → Retrieved from **both** memories
3. After 3+ interactions → Session summary saved to **long-term**
4. Future sessions → Can access **all** past knowledge

---

## 📂 Project Structure

```
Research_Assistant_AI/
├── src/                          # Core source code
│   ├── graph.py                  # LangGraph workflow (12 nodes)
│   ├── state.py                  # State definitions (TypedDict)
│   ├── configuration.py          # Pydantic configuration
│   └── utils/                    # Utility modules
│       ├── pdf_utils.py          # PDF processing (PyMuPDF, pdfplumber)
│       ├── embedding_utils.py    # Sentence Transformers wrapper
│       ├── vector_db_utils.py    # ChromaDB operations
│       └── session_utils.py      # Session cache management
├── app.py                        # Streamlit web UI (400+ lines)
├── cli.py                        # CLI interface (400+ lines)
├── data/                         # Data storage
│   ├── chromadb/                 # Vector database (persistent)
│   └── sessions/                 # Session cache (JSON)
├── logs/                         # Application logs
│   ├── app.log                   # Web UI logs
│   └── cli.log                   # CLI logs
├── uploads/                      # Temporary PDF storage
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment configuration template
├── QUICKSTART.md                 # Quick start tutorial
├── CLI_DEBUG_GUIDE.md           # CLI debugging documentation
└── README.md                     # This file
```

---

## 💻 Usage

### Web UI (Streamlit)

**Best for**: Regular use, non-technical users, production deployment

```powershell
streamlit run app.py
```

**Features**:
- 📤 Drag-and-drop PDF upload
- 💬 Chat-style interface
- 📊 Session statistics sidebar
- 📚 Source display with relevance scores
- 🎨 Clean, intuitive design

**Workflow**:
1. Upload PDF via sidebar
2. Wait for processing confirmation
3. Type question in chat input
4. View answer with sources
5. Ask follow-up questions

### CLI (Command-Line Interface)

**Best for**: Development, debugging, automation, Copilot agent mode

```powershell
python cli.py
```

**Commands**:
| Command | Description |
|---------|-------------|
| `/upload <path>` | Upload and process PDF |
| `/list` | List uploaded PDFs |
| `/session` | Show session info |
| `/debug on\|off` | Toggle debug logging |
| `/clear` | Clear session |
| `/help` | Show help |
| `/exit` | Exit CLI |

**Example Session**:
```
>>> /debug on
>>> /upload "D:\Papers\research_paper.pdf"
✅ PDF processed: research_paper.pdf (152 chunks)

>>> What is the main hypothesis?
💡 Answer:
----------------------------------------------------------------------
The paper hypothesizes that transformer models with self-attention...
----------------------------------------------------------------------

📚 Sources (3):
   1. 🟢 Relevance: 0.923 | Page 3 | Section: Introduction
   2. 🟡 Relevance: 0.847 | Page 5 | Section: Methodology
   3. 🟢 Relevance: 0.901 | Page 12 | Section: Results

📊 Interaction: 1
```

📖 **Detailed CLI guide**: See [CLI_DEBUG_GUIDE.md](CLI_DEBUG_GUIDE.md)

---

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
# LLM Configuration
LOCAL_LLM=qwen2.5:7b              # Ollama model name
OLLAMA_BASE_URL=http://localhost:11434
TEMPERATURE=0.0                    # 0.0 = deterministic

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Sentence transformer
EMBEDDING_DIMENSION=384            # Must match model

# Vector Database
VECTOR_DB_TYPE=chromadb
CHROMADB_PATH=./data/chromadb

# PDF Processing
PDF_CHUNK_SIZE=1000               # Characters per chunk
PDF_CHUNK_OVERLAP=200             # Overlap for context

# Retrieval
TOP_K_SHORT_TERM=5                # Chunks from session
TOP_K_LONG_TERM=3                 # Chunks from vector DB
SIMILARITY_THRESHOLD=0.5          # Min relevance score

# Session
PERSIST_THRESHOLD=3               # Interactions before saving

# Logging
LOG_LEVEL=INFO                    # DEBUG|INFO|WARNING|ERROR
```

### Pydantic Configuration

All settings are validated by `ResearchAgentConfig` in `src/configuration.py`:

```python
config = ResearchAgentConfig(
    local_llm="qwen2.5:7b",
    temperature=0.0,
    embedding_model="all-MiniLM-L6-v2",
    top_k_short_term=5,
    persist_threshold=3
)
```

Can be overridden via `RunnableConfig`:

```python
from langchain_core.runnables import RunnableConfig

runnable_config = RunnableConfig(
    configurable={
        "temperature": 0.7,
        "top_k_short_term": 10
    }
)

result = graph.invoke(state, config=runnable_config)
```

---

## 🔧 Development

### Prerequisites

- Python 3.11+
- Git
- Ollama
- VS Code (recommended)

### Setup Development Environment

1. **Clone and setup**
   ```powershell
   git clone https://github.com/Sabit166/Research_Assistant_AI.git
   cd Research_Assistant_AI
   python -m venv env
   .\env\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Enable debug logging**
   ```powershell
   # In .env
   LOG_LEVEL=DEBUG
   DEBUG=true
   LANGCHAIN_DEBUG=true
   ```

3. **Run tests**
   ```powershell
   python cli.py
   >>> /debug on
   >>> /upload "test_document.pdf"
   ```

### Project Guidelines

- **Code Style**: PEP 8, type hints required
- **Documentation**: Docstrings for all functions
- **Logging**: Use `logger.debug()` for detailed traces
- **Testing**: Test via CLI with `/debug on`
- **State Management**: Always use TypedDict schemas

### Debugging Tips

1. **Use CLI for debugging**: Better logging than Web UI
2. **Enable debug mode**: `/debug on` in CLI
3. **Check logs**: `logs/cli.log` has full traces
4. **Test nodes individually**: Modify `graph.py` to isolate nodes
5. **Verify ChromaDB**: Check `data/chromadb/` for .bin files

📖 **Full debugging guide**: [CLI_DEBUG_GUIDE.md](CLI_DEBUG_GUIDE.md)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - project overview |
| [QUICKSTART.md](QUICKSTART.md) | Step-by-step setup and first use |
| [CLI_DEBUG_GUIDE.md](CLI_DEBUG_GUIDE.md) | CLI commands and debugging workflows |
| [.env.example](.env.example) | Environment configuration reference |

### Code Documentation

- **src/graph.py**: Complete LangGraph workflow with 12 nodes
- **src/state.py**: State schema definitions (ResearchState)
- **src/configuration.py**: Pydantic configuration model
- **src/utils/**: Utility modules (PDF, embeddings, vector DB, sessions)

---

## 🎯 Use Cases

### Academic Research
- Upload research papers and textbooks
- Ask synthesis questions across multiple papers
- Build long-term knowledge base for your field

### Documentation Analysis
- Process technical documentation (API docs, manuals)
- Quick reference for complex systems
- Compare versions and track changes

### Legal & Compliance
- Analyze contracts and legal documents
- Find specific clauses and definitions
- Cross-reference multiple documents

### Business Intelligence
- Process reports and presentations
- Extract insights from business documents
- Build organizational knowledge base

---

## 🚀 Performance

### Expected Processing Times

| Task | Small (5p) | Medium (50p) | Large (500p) |
|------|-----------|--------------|--------------|
| PDF Upload | ~10s | ~30s | ~5min |
| First Query | ~5s | ~5s | ~5s |
| Follow-up | ~3s | ~3s | ~3s |

*Tested on: Intel i7, 16GB RAM, no GPU*

### Resource Usage

- **RAM**: 2-4 GB (model + embeddings)
- **Disk**: ~500 MB per 100-page PDF
- **CPU**: 20-50% during processing

### Optimization Tips

1. **Use smaller models**: `ollama pull qwen2.5:3b`
2. **Reduce chunk size**: Set `PDF_CHUNK_SIZE=500`
3. **Limit retrieval**: Set `TOP_K_SHORT_TERM=3`
4. **Clean old sessions**: Delete `data/sessions/*.json`

---

## 🛠️ Troubleshooting

### Common Issues

<details>
<summary><b>Cannot connect to Ollama</b></summary>

**Error**: `Connection refused to localhost:11434`

**Solution**:
```powershell
ollama serve
```
Keep terminal open while using the app.

</details>

<details>
<summary><b>Model not found</b></summary>

**Error**: `Model qwen2.5:7b not found`

**Solution**:
```powershell
ollama pull qwen2.5:7b
```
Wait for download (~4GB).

</details>

<details>
<summary><b>PDF upload fails</b></summary>

**Error**: PDF processes but no text extracted

**Causes**:
- Scanned PDF (image, no text layer)
- Encrypted PDF
- Corrupted file

**Solutions**:
- Use OCR tool first
- Unlock PDF with password
- Try different PDF

</details>

<details>
<summary><b>Out of memory errors</b></summary>

**Error**: `Out of memory` or system freezes

**Solutions**:
- Close other applications
- Use smaller model: `qwen2.5:3b`
- Process smaller PDFs (<50 pages)
- Increase system swap/pagefile

</details>

<details>
<summary><b>Slow performance</b></summary>

**Causes**:
- First model load (slow once)
- Large PDFs
- Limited RAM
- Many background processes

**Solutions**:
- Wait for first query (loads model)
- Process in chunks
- Close browser tabs
- Use CLI instead of Web UI

</details>

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Priorities

- [ ] Add support for other document types (DOCX, TXT, HTML)
- [ ] Implement streaming responses in Web UI
- [ ] Add user authentication and multi-user support
- [ ] Create REST API for programmatic access
- [ ] Add support for cloud LLMs (OpenAI, Anthropic)
- [ ] Implement advanced RAG techniques (HyDE, query expansion)
- [ ] Add visualization of document relationships
- [ ] Create Docker deployment option

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with amazing open-source technologies:

- **[LangChain](https://github.com/langchain-ai/langchain)** - Framework for LLM applications
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - State machine orchestration
- **[Ollama](https://ollama.ai/)** - Local LLM runtime
- **[ChromaDB](https://www.trychroma.com/)** - Vector database
- **[Sentence Transformers](https://www.sbert.net/)** - Embedding models
- **[Streamlit](https://streamlit.io/)** - Web UI framework
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** - PDF processing
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation

---

## 📞 Contact

**Repository**: [github.com/Sabit166/Research_Assistant_AI](https://github.com/Sabit166/Research_Assistant_AI)

**Issues**: [github.com/Sabit166/Research_Assistant_AI/issues](https://github.com/Sabit166/Research_Assistant_AI/issues)

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ using LangGraph and Ollama

</div>
