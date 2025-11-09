# Research Assistant AI Agent with Memory

A sophisticated AI research assistant built with LangGraph that features dual memory systems (short-term session memory + long-term vector database) and PDF document processing capabilities.

## 🎯 Features

### Core Capabilities
- **PDF Processing**: Upload and process PDF documents with intelligent chunking
- **Dual Memory System**:
  - **Short-term Memory**: Session-based conversation history and document cache
  - **Long-term Memory**: Persistent vector database for historical knowledge
- **Intelligent Retrieval**: Query both memory systems simultaneously
- **Context Merging**: Deduplicate and rank results from multiple sources
- **Local LLM Support**: Works with Ollama or LM Studio for privacy
- **Session Persistence**: Automatic summarization and storage of valuable sessions

### Architecture Highlights
- **12-Node LangGraph**: Sophisticated workflow with conditional routing
- **Vector Embeddings**: sentence-transformers for semantic search
- **Multiple Vector DBs**: ChromaDB (local) or Pinecone (cloud)
- **Configurable**: Environment variables and runtime configuration

## 📋 Architecture

```
START → session_init → check_input 
          ↓                ↓
      [PDF Upload?]    [Query?]
          ↓                ↓
      ingest_pdf ← → process_query
          ↓                ↓
    confirm_ingest   retrieve_memory
          ↓                ↓
          └────→ merge_context ←────┘
                     ↓
              generate_answer
                     ↓
              update_session
                     ↓
              should_persist?
                ↓         ↓
    persist_to_longterm  finalize_response
                ↓         ↓
                └────→ END ←────┘
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Ollama or LM Studio for local LLM
- Git

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd Research_Assistant_AI
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Setting up Local LLM

#### Using Ollama (Recommended)
1. Download Ollama from https://ollama.com/download
2. Pull a model:
```bash
ollama pull llama3.2
```
3. The server starts automatically on `http://localhost:11434`

#### Using LM Studio
1. Download LM Studio from https://lmstudio.ai/
2. Load a model (e.g., qwen_qwq-32b)
3. Start the local server in the "Local Server" tab
4. Note the URL (default: `http://localhost:1234/v1`)

## 📖 Usage

### Basic Usage

```python
from src.graph import graph
from langchain_core.runnables import RunnableConfig

# Example 1: Upload a PDF
result = graph.invoke(
    {
        "user_query": "",
        "uploaded_pdfs": ["path/to/document.pdf"]
    },
    config=RunnableConfig(configurable={
        "llm_provider": "ollama",
        "local_llm": "llama3.2"
    })
)
print(result["answer"])

# Example 2: Ask a question
result = graph.invoke(
    {
        "user_query": "What are the main topics in the uploaded documents?",
        "uploaded_pdfs": None
    },
    config=RunnableConfig(configurable={
        "llm_provider": "ollama",
        "local_llm": "llama3.2"
    })
)
print(result["answer"])
print("Sources:", result["sources"])
```

### Using LangGraph Studio

1. **Install LangGraph CLI**:
```bash
pip install -U "langgraph-cli[inmem]"
```

2. **Start the server**:
```bash
langgraph dev
```

3. **Open LangGraph Studio**:
Navigate to `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

4. **Configure in the UI**:
- Set LLM provider, model, and other settings in the Configuration tab
- Upload PDFs or ask questions in the Input tab
- View the graph execution in real-time

## ⚙️ Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# LLM Configuration
LLM_PROVIDER=ollama
LOCAL_LLM=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Database
VECTOR_DB_TYPE=chromadb
CHROMADB_PATH=./data/chromadb

# Retrieval Settings
TOP_K_SHORT_TERM=5
TOP_K_LONG_TERM=3

# Persistence
PERSIST_THRESHOLD=3
```

### Runtime Configuration

Configure via `RunnableConfig`:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(configurable={
    "llm_provider": "ollama",
    "local_llm": "llama3.2",
    "top_k_short_term": 5,
    "top_k_long_term": 3,
    "persist_threshold": 3
})

result = graph.invoke(input_data, config=config)
```

**Configuration Priority**:
1. Environment variables (highest)
2. Runtime `RunnableConfig`
3. Default values in `ResearchAgentConfig` (lowest)

## 🗂️ Project Structure

```
Research_Assistant_AI/
├── src/
│   ├── graph.py              # Main graph implementation (12 nodes)
│   ├── state.py              # State definitions
│   ├── configuration.py      # Configuration schema
│   └── utils/
│       ├── __init__.py
│       ├── pdf_utils.py      # PDF extraction and chunking
│       ├── embedding_utils.py # Embedding generation
│       ├── vector_db_utils.py # Vector database clients
│       └── session_utils.py  # Session management
├── ollama_deep_researcher/   # Example: Web research agent
├── data/                      # Data directory (auto-created)
│   ├── chromadb/             # ChromaDB persistence
│   └── sessions/             # Session cache
├── architecture.txt          # Architecture diagram
├── requirements.txt          # Python dependencies
├── .env.example              # Example environment config
└── README.md                 # This file
```

## 🔧 Node Descriptions

### 1. **session_init**
Initializes a unique session ID and clears short-term memory.

### 2. **check_input** (Conditional)
Determines if the input is a PDF upload or a query and routes accordingly.

### 3. **ingest_pdf**
- Extracts text from PDFs
- Chunks text with overlap
- Generates embeddings
- Stores in both session memory and vector DB

### 4. **confirm_ingest**
Sends confirmation message to user about PDF processing.

### 5. **process_query**
Parses and normalizes the user's query.

### 6. **retrieve_memory**
- Retrieves from short-term session memory (semantic search)
- Retrieves from long-term vector database
- Returns separate result sets with scores

### 7. **merge_context**
- Combines short-term and long-term results
- Deduplicates similar content
- Ranks by relevance score
- Limits by token budget

### 8. **generate_answer**
- Builds prompt with query and merged context
- Calls LLM to generate answer
- Extracts sources used in the answer

### 9. **update_session**
- Adds query and answer to session messages
- Updates conversation history
- Saves session to cache

### 10. **should_persist** (Conditional)
Checks if session should be persisted to long-term memory based on interaction count.

### 11. **persist_to_longterm**
- Summarizes the session
- Generates embedding for summary
- Stores in global long-term collection

### 12. **finalize_response**
Formats final response with answer, sources, and metadata.

## 🔍 Memory Systems

### Short-term Memory (Session)
- **Lifetime**: Current session only
- **Storage**: In-memory + session cache files
- **Purpose**: Recent conversation history and uploaded documents
- **Retrieval**: Semantic similarity search on session documents

### Long-term Memory (Vector DB)
- **Lifetime**: Persistent across sessions
- **Storage**: ChromaDB or Pinecone
- **Purpose**: Historical knowledge from past sessions
- **Retrieval**: Vector similarity search on session summaries

### Memory Persistence
Sessions are automatically persisted when:
- Interaction count ≥ `PERSIST_THRESHOLD` (default: 3)
- Session contains valuable information
- User explicitly saves the session

## 📊 Dependencies

### Core
- `langgraph>=0.2.0` - Graph orchestration
- `langchain>=0.3.0` - LLM framework
- `langchain-ollama>=0.2.0` - Ollama integration

### PDF Processing
- `pymupdf>=1.24.0` - Fast PDF extraction
- `pdfplumber>=0.11.0` - Alternative PDF extraction

### Embeddings & Vector DB
- `sentence-transformers>=2.2.0` - Embedding generation
- `chromadb>=0.4.0` - Local vector database
- `pinecone-client>=3.0.0` - Cloud vector database (optional)

### Utilities
- `pydantic>=2.0.0` - Configuration validation
- `numpy>=1.24.0` - Vector operations

## 🧪 Testing

```bash
# Run basic test
python -c "from src.graph import graph; print('Graph loaded successfully!')"

# Test PDF processing
python -c "from src.utils import process_pdf; print(process_pdf('test.pdf'))"

# Test embeddings
python -c "from src.utils import EmbeddingGenerator; gen = EmbeddingGenerator(); print(len(gen.generate_embedding('test')))"
```

## 🐛 Troubleshooting

### Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt
```

### PDF Processing Errors
```bash
# Install PDF libraries
pip install pymupdf pdfplumber
```

### ChromaDB Errors
```bash
# Clear ChromaDB cache
rm -rf data/chromadb
```

### LLM Connection Errors
- Ensure Ollama/LM Studio is running
- Check the base URL in `.env`
- Verify the model is pulled: `ollama list`

## 📝 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

[Your Contact Information]

---

**Built with ❤️ using LangGraph, Ollama, and ChromaDB**
