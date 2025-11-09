"""Configuration for the Research Assistant Agent."""
import os
from dataclasses import dataclass, field
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig


class ResearchAgentConfig(BaseModel):
    """Configuration schema for Research Assistant."""
    
    # LLM Configuration
    local_llm: str = Field(
        default="llama3.2",
        description="Name of the local LLM model (Ollama)"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama service"
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature for generation"
    )
    
    # Embedding Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for embeddings"
    )
    embedding_dimension: int = Field(
        default=384,
        description="Dimension of embedding vectors"
    )
    
    # Vector Database Configuration
    vector_db_type: Literal["chromadb", "pinecone"] = Field(
        default="chromadb",
        description="Type of vector database to use"
    )
    chromadb_path: str = Field(
        default="./data/chromadb",
        description="Path for ChromaDB persistence"
    )
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="API key for Pinecone (if using)"
    )
    pinecone_environment: Optional[str] = Field(
        default=None,
        description="Pinecone environment (if using)"
    )
    pinecone_index_name: str = Field(
        default="research-assistant",
        description="Pinecone index name"
    )
    
    # PDF Processing Configuration
    pdf_chunk_size: int = Field(
        default=1000,
        description="Size of text chunks for PDF processing"
    )
    pdf_chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks"
    )
    
    # Retrieval Configuration
    top_k_short_term: int = Field(
        default=5,
        description="Number of results to retrieve from short-term memory"
    )
    top_k_long_term: int = Field(
        default=3,
        description="Number of results to retrieve from long-term memory"
    )
    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum similarity score for retrieval"
    )
    
    # Context Configuration
    max_context_tokens: int = Field(
        default=4000,
        description="Maximum tokens for merged context"
    )
    
    # Persistence Configuration
    persist_threshold: int = Field(
        default=3,
        description="Number of interactions before persisting to long-term memory"
    )
    session_cache_dir: str = Field(
        default="./data/sessions",
        description="Directory for session cache files"
    )
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "ResearchAgentConfig":
        """Create configuration from RunnableConfig and environment variables.
        
        Priority order:
        1. Environment variables (highest)
        2. RunnableConfig configurable dict
        3. Default values in the class (lowest)
        """
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        
        # Build kwargs from environment variables (highest priority)
        kwargs = {}
        
        # LLM settings
        if local_llm := os.environ.get("LOCAL_LLM"):
            kwargs["local_llm"] = local_llm
        elif "local_llm" in configurable:
            kwargs["local_llm"] = configurable["local_llm"]
            
        if ollama_base_url := os.environ.get("OLLAMA_BASE_URL"):
            kwargs["ollama_base_url"] = ollama_base_url
        elif "ollama_base_url" in configurable:
            kwargs["ollama_base_url"] = configurable["ollama_base_url"]
            
        # Embedding settings
        if embedding_model := os.environ.get("EMBEDDING_MODEL"):
            kwargs["embedding_model"] = embedding_model
        elif "embedding_model" in configurable:
            kwargs["embedding_model"] = configurable["embedding_model"]
            
        # Vector DB settings
        if vector_db_type := os.environ.get("VECTOR_DB_TYPE"):
            kwargs["vector_db_type"] = vector_db_type
        elif "vector_db_type" in configurable:
            kwargs["vector_db_type"] = configurable["vector_db_type"]
            
        if chromadb_path := os.environ.get("CHROMADB_PATH"):
            kwargs["chromadb_path"] = chromadb_path
        elif "chromadb_path" in configurable:
            kwargs["chromadb_path"] = configurable["chromadb_path"]
            
        if pinecone_api_key := os.environ.get("PINECONE_API_KEY"):
            kwargs["pinecone_api_key"] = pinecone_api_key
            
        # Retrieval settings
        if top_k_short_term := os.environ.get("TOP_K_SHORT_TERM"):
            kwargs["top_k_short_term"] = int(top_k_short_term)
        elif "top_k_short_term" in configurable:
            kwargs["top_k_short_term"] = configurable["top_k_short_term"]
            
        if top_k_long_term := os.environ.get("TOP_K_LONG_TERM"):
            kwargs["top_k_long_term"] = int(top_k_long_term)
        elif "top_k_long_term" in configurable:
            kwargs["top_k_long_term"] = configurable["top_k_long_term"]
            
        # Persistence settings
        if persist_threshold := os.environ.get("PERSIST_THRESHOLD"):
            kwargs["persist_threshold"] = int(persist_threshold)
        elif "persist_threshold" in configurable:
            kwargs["persist_threshold"] = configurable["persist_threshold"]
        
        return cls(**kwargs)
