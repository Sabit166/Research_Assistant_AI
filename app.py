"""Streamlit Web UI for Research Assistant AI.

This is the main entry point for the Research Assistant AI application.
It provides a user-friendly web interface for uploading PDFs and asking questions.
"""
import sys
from pathlib import Path
import streamlit as st
import logging
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from graph import graph
    from configuration import ResearchAgentConfig
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")
    st.error("Please ensure all required files are in the 'src' directory.")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        Path("./uploads"),
        Path("./data/sessions"),
        Path("./data/chromadb"),
        Path("./logs")
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            st.error(f"❌ Could not create directory: {directory}")


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0
    
    if "uploaded_pdfs" not in st.session_state:
        st.session_state.uploaded_pdfs = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False


def process_pdf_upload(uploaded_file):
    """Process uploaded PDF file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        tuple: (success: bool, message: str, result: dict or None)
    """
    try:
        # Save uploaded file to uploads directory
        upload_path = Path("./uploads") / uploaded_file.name
        
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"Saved uploaded file: {upload_path}")
        
        # Process PDF with graph
        config = {"configurable": ResearchAgentConfig().model_dump()}
        
        result = graph.invoke({
            "uploaded_pdfs": [str(upload_path)],
            "user_query": ""
        }, config=config)
        
        # Update session state
        st.session_state.session_id = result.get("session_id")
        if str(upload_path) not in st.session_state.uploaded_pdfs:
            st.session_state.uploaded_pdfs.append(str(upload_path))
        
        logger.info(f"Successfully processed PDF: {uploaded_file.name}")
        
        return True, result.get("answer", "PDF processed successfully!"), result
        
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg, None


def process_query(query: str):
    """Process user query through the graph.
    
    Args:
        query: User's question
        
    Returns:
        tuple: (success: bool, result: dict or None, error_msg: str)
    """
    try:
        config = {"configurable": ResearchAgentConfig().model_dump()}
        
        result = graph.invoke({
            "uploaded_pdfs": [],
            "user_query": query
        }, config=config)
        
        # Update session state
        if not st.session_state.session_id:
            st.session_state.session_id = result.get("session_id")
        
        st.session_state.interaction_count = result.get("interaction_count", 0)
        
        logger.info(f"Successfully processed query: {query[:50]}...")
        
        return True, result, ""
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


def display_sources(sources):
    """Display sources in a formatted way.
    
    Args:
        sources: List of source dictionaries
    """
    if not sources:
        st.info("No sources available for this answer.")
        return
    
    st.markdown("### 📚 Sources")
    
    # Group by memory type
    short_term = [s for s in sources if s.get("memory_type") == "short_term"]
    long_term = [s for s in sources if s.get("memory_type") == "long_term"]
    
    if short_term:
        st.markdown("**Current Session Documents:**")
        for i, source in enumerate(short_term[:5], 1):
            score = source.get("score", 0)
            source_name = source.get("source", "Unknown")
            chunk_id = source.get("chunk_id", "")
            
            # Color code by relevance
            if score >= 0.9:
                color = "🟢"
            elif score >= 0.7:
                color = "🟡"
            else:
                color = "🔴"
            
            st.markdown(
                f"{color} **[{i}]** {source_name} "
                f"(relevance: {score:.2%})"
            )
            
            # Show text preview in expander
            if source.get("text"):
                with st.expander(f"Preview {i}"):
                    st.text(source["text"][:300] + "...")
    
    if long_term:
        st.markdown("**Previous Sessions:**")
        for i, source in enumerate(long_term[:3], 1):
            score = source.get("score", 0)
            session_id = source.get("session_id", "Unknown")
            timestamp = source.get("timestamp", "")
            
            st.markdown(
                f"🔵 **[{i}]** Session: {session_id[:16]}... "
                f"(relevance: {score:.2%})"
            )
            
            if source.get("text"):
                with st.expander(f"Summary {i}"):
                    st.text(source["text"][:300] + "...")


def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="Research Assistant AI",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Ensure directories exist
    ensure_directories()
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🤖 Research Assistant AI")
    st.markdown(
        "*Your AI-powered research companion with dual memory system "
        "(short-term session + long-term knowledge base)*"
    )
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Document Management")
        
        # PDF Upload Section
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to analyze and ask questions about"
        )
        
        if uploaded_file is not None:
            st.info(f"📎 Selected: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            if st.button("🚀 Process PDF", type="primary", use_container_width=True):
                with st.spinner("⏳ Processing PDF... This may take a minute."):
                    success, message, result = process_pdf_upload(uploaded_file)
                    
                    if success:
                        st.success("✅ " + message)
                        # Clear the file uploader by rerunning
                        st.rerun()
                    else:
                        st.error("❌ " + message)
        
        st.divider()
        
        # Uploaded PDFs List
        st.subheader("📚 Uploaded Documents")
        if st.session_state.uploaded_pdfs:
            for i, pdf_path in enumerate(st.session_state.uploaded_pdfs, 1):
                pdf_name = Path(pdf_path).name
                st.markdown(f"**{i}.** {pdf_name}")
        else:
            st.info("No documents uploaded yet.")
        
        st.divider()
        
        # Session Information
        st.subheader("📊 Session Info")
        
        if st.session_state.session_id:
            session_display = st.session_state.session_id[:20] + "..."
            st.text_input(
                "Session ID:",
                value=session_display,
                disabled=True,
                help="Your unique session identifier"
            )
        else:
            st.info("No active session")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            st.metric("Interactions", st.session_state.interaction_count)
        
        # Persistence status
        if st.session_state.interaction_count >= 3:
            st.success("✅ Session persisted to long-term memory")
        elif st.session_state.interaction_count > 0:
            remaining = 3 - st.session_state.interaction_count
            st.info(f"📝 {remaining} more interaction(s) until persistence")
        
        st.divider()
        
        # Clear Chat Button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.interaction_count = 0
            st.rerun()
        
        # System Info
        with st.expander("ℹ️ System Information"):
            st.markdown("""
            **Features:**
            - 📄 PDF text extraction & chunking
            - 🧠 Dual memory system
            - 🔍 Semantic similarity search
            - 💾 Persistent long-term knowledge
            - 🤖 Local LLM (Ollama)
            
            **Memory Types:**
            - Short-term: Current session
            - Long-term: Cross-session knowledge
            """)
    
    # Main Chat Area
    st.header("💬 Chat Interface")
    
    # Display welcome message if no messages
    if not st.session_state.messages:
        st.info("""
        👋 **Welcome to Research Assistant AI!**
        
        **Getting Started:**
        1. 📄 Upload a PDF document using the sidebar
        2. 💬 Ask questions about the content
        3. 🧠 Build your knowledge base across sessions
        
        **Tips:**
        - You can upload multiple PDFs
        - Your conversation is saved after 3+ interactions
        - Previous session knowledge helps answer future questions
        """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message.get("sources") and message["role"] == "assistant":
                with st.expander(f"📚 View Sources ({len(message['sources'])})"):
                    display_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your documents...",
        disabled=st.session_state.processing
    ):
        # Validation
        if not prompt.strip():
            st.warning("⚠️ Please enter a valid question.")
            st.stop()
        
        if not st.session_state.uploaded_pdfs:
            st.warning("⚠️ Please upload at least one PDF document first.")
            st.stop()
        
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                st.session_state.processing = True
                
                success, result, error_msg = process_query(prompt)
                
                st.session_state.processing = False
                
                if success:
                    answer = result.get("answer", "I couldn't generate an answer.")
                    sources = result.get("sources", [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander(f"📚 View Sources ({len(sources)})"):
                            display_sources(sources)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                else:
                    error_display = f"❌ {error_msg}"
                    st.error(error_display)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_display,
                        "sources": []
                    })
        
        # Rerun to update UI
        st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        st.error(f"❌ Fatal error: {e}")
        st.error("Please check the logs for details.")
