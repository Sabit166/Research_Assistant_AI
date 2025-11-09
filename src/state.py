"""State definitions for the Research Assistant Agent."""
from typing import TypedDict, Annotated, Sequence, Optional
from langgraph.graph import add_messages


class ResearchState(TypedDict):
    """Complete state for the Research Assistant agent."""
    
    # Input fields
    user_query: str
    uploaded_pdfs: Optional[list[str]]  # List of PDF file paths
    
    # Session management
    session_id: str
    session_messages: Annotated[Sequence[dict], add_messages]  # Short-term conversation history
    session_docs: list[dict]  # Short-term document chunks from current session
    
    # Processing state
    input_type: str  # "pdf_upload" or "query"
    parsed_query: str
    
    # Retrieval results
    retrieved_short_term: list[dict]  # From session memory
    retrieved_long_term: list[dict]  # From vector DB
    merged_context: list[dict]  # Combined and ranked results
    
    # Output fields
    answer: str
    sources: list[dict]  # Sources used in the answer
    
    # Control flow
    should_persist: bool  # Whether to save session to long-term memory
    
    # Metadata
    pdf_chunks_count: int
    retrieval_scores: dict
    timestamp: str


class ResearchStateInput(TypedDict):
    """Input schema for the Research Assistant."""
    user_query: str
    uploaded_pdfs: Optional[list[str]]


class ResearchStateOutput(TypedDict):
    """Output schema for the Research Assistant."""
    answer: str
    sources: list[dict]
    metadata: dict
