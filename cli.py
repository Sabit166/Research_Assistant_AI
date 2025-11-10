"""CLI Interface for Research Assistant AI.

A command-line interface for development, testing, and debugging.
Provides interactive access to the research assistant with detailed logging.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph import create_research_graph
from src.configuration import ResearchAgentConfig
from src.state import ResearchState

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cli.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ResearchAssistantCLI:
    """Command-line interface for Research Assistant AI."""
    
    def __init__(self):
        """Initialize CLI."""
        self.setup_directories()
        self.config = ResearchAgentConfig()
        self.graph = create_research_graph(self.config)
        self.session_id = self._generate_session_id()
        self.interaction_count = 0
        self.uploaded_pdfs = []
        
        logger.info(f"CLI initialized with session: {self.session_id}")
    
    def setup_directories(self):
        """Create required directories."""
        dirs = ['uploads', 'data/sessions', 'data/chromadb', 'logs']
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def print_banner(self):
        """Print welcome banner."""
        print("\n" + "=" * 70)
        print("🤖 Research Assistant AI - CLI Interface")
        print("=" * 70)
        print(f"Session ID: {self.session_id}")
        print(f"Model: {self.config.model_name}")
        print(f"Embeddings: {self.config.embedding_model}")
        print("=" * 70)
        print("\nCommands:")
        print("  /upload <filepath>  - Upload and process a PDF")
        print("  /list              - List uploaded PDFs")
        print("  /session           - Show session info")
        print("  /clear             - Clear session and start fresh")
        print("  /debug on|off      - Toggle debug logging")
        print("  /help              - Show this help")
        print("  /exit or /quit     - Exit the CLI")
        print("\nJust type your question to query the research assistant.")
        print("=" * 70 + "\n")
    
    def process_command(self, command: str) -> bool:
        """
        Process CLI commands.
        
        Args:
            command: Command string starting with /
            
        Returns:
            True to continue, False to exit
        """
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None
        
        if cmd in ['/exit', '/quit']:
            print("\n👋 Goodbye! Session saved.")
            return False
        
        elif cmd == '/help':
            self.print_banner()
        
        elif cmd == '/upload':
            if not arg:
                print("❌ Usage: /upload <filepath>")
            else:
                self.upload_pdf(arg)
        
        elif cmd == '/list':
            self.list_pdfs()
        
        elif cmd == '/session':
            self.show_session_info()
        
        elif cmd == '/clear':
            self.clear_session()
        
        elif cmd == '/debug':
            self.toggle_debug(arg)
        
        else:
            print(f"❌ Unknown command: {cmd}")
            print("Type /help for available commands")
        
        return True
    
    def upload_pdf(self, filepath: str):
        """Upload and process a PDF."""
        path = Path(filepath)
        
        if not path.exists():
            print(f"❌ File not found: {filepath}")
            return
        
        if path.suffix.lower() != '.pdf':
            print(f"❌ Not a PDF file: {filepath}")
            return
        
        print(f"\n📄 Processing PDF: {path.name}")
        print("   Please wait...")
        
        try:
            # Create initial state
            initial_state: ResearchState = {
                "messages": [],
                "session_id": self.session_id,
                "current_query": "",
                "pdf_content": "",
                "pdf_file_path": str(path.absolute()),
                "relevant_chunks": [],
                "session_memory": {},
                "longterm_memory": [],
                "interaction_count": self.interaction_count,
                "should_persist": False,
                "current_step": "ingest_pdf",
                "error": None,
                "next_action": "ingest_pdf"
            }
            
            # Run graph
            result = self.graph.invoke(initial_state)
            
            if result.get("error"):
                print(f"❌ Error: {result['error']}")
            else:
                self.uploaded_pdfs.append({
                    'name': path.name,
                    'path': str(path.absolute()),
                    'size': path.stat().st_size,
                    'timestamp': datetime.now().isoformat()
                })
                print(f"✅ PDF processed successfully: {path.name}")
                print(f"   Size: {path.stat().st_size / 1024:.1f} KB")
        
        except Exception as e:
            logger.error(f"Error processing PDF: {e}", exc_info=True)
            print(f"❌ Error processing PDF: {e}")
    
    def list_pdfs(self):
        """List uploaded PDFs."""
        if not self.uploaded_pdfs:
            print("\n📚 No PDFs uploaded yet")
            print("   Use /upload <filepath> to add documents")
        else:
            print(f"\n📚 Uploaded PDFs ({len(self.uploaded_pdfs)}):")
            for i, pdf in enumerate(self.uploaded_pdfs, 1):
                print(f"   {i}. {pdf['name']}")
                print(f"      Size: {pdf['size'] / 1024:.1f} KB")
                print(f"      Time: {pdf['timestamp']}")
    
    def show_session_info(self):
        """Show session information."""
        print(f"\n📊 Session Information:")
        print(f"   Session ID: {self.session_id}")
        print(f"   Interactions: {self.interaction_count}")
        print(f"   PDFs Uploaded: {len(self.uploaded_pdfs)}")
        print(f"   Model: {self.config.model_name}")
        print(f"   Temperature: {self.config.temperature}")
        print(f"   Persist Threshold: {self.config.persist_threshold}")
    
    def clear_session(self):
        """Clear session and start fresh."""
        confirm = input("\n⚠️  Clear all session data? (yes/no): ").strip().lower()
        if confirm == 'yes':
            self.session_id = self._generate_session_id()
            self.interaction_count = 0
            self.uploaded_pdfs = []
            print(f"✅ Session cleared. New session: {self.session_id}")
        else:
            print("❌ Clear cancelled")
    
    def toggle_debug(self, mode: Optional[str]):
        """Toggle debug logging."""
        if not mode:
            print("❌ Usage: /debug on|off")
            return
        
        if mode.lower() == 'on':
            logging.getLogger().setLevel(logging.DEBUG)
            print("✅ Debug logging enabled")
        elif mode.lower() == 'off':
            logging.getLogger().setLevel(logging.INFO)
            print("✅ Debug logging disabled")
        else:
            print("❌ Use: /debug on or /debug off")
    
    def process_query(self, query: str):
        """
        Process a user query.
        
        Args:
            query: User's question
        """
        if not query.strip():
            return
        
        print(f"\n🤔 Processing query...")
        logger.info(f"Query: {query}")
        
        try:
            # Increment interaction count
            self.interaction_count += 1
            
            # Create state
            initial_state: ResearchState = {
                "messages": [],
                "session_id": self.session_id,
                "current_query": query,
                "pdf_content": "",
                "pdf_file_path": "",
                "relevant_chunks": [],
                "session_memory": {},
                "longterm_memory": [],
                "interaction_count": self.interaction_count,
                "should_persist": False,
                "current_step": "process_query",
                "error": None,
                "next_action": "process_query"
            }
            
            # Run graph
            result = self.graph.invoke(initial_state)
            
            # Check for errors
            if result.get("error"):
                print(f"\n❌ Error: {result['error']}")
                logger.error(f"Graph error: {result['error']}")
                return
            
            # Extract answer from messages
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    answer = last_message.content
                else:
                    answer = str(last_message)
                
                print(f"\n💡 Answer:")
                print("-" * 70)
                print(answer)
                print("-" * 70)
            
            # Show relevant sources
            relevant_chunks = result.get("relevant_chunks", [])
            if relevant_chunks:
                print(f"\n📚 Sources ({len(relevant_chunks)}):")
                for i, chunk in enumerate(relevant_chunks[:3], 1):  # Show top 3
                    score = chunk.get('score', 0.0)
                    content = chunk.get('content', '')[:100] + "..."
                    metadata = chunk.get('metadata', {})
                    
                    # Color code by relevance
                    if score >= 0.9:
                        indicator = "🟢"
                    elif score >= 0.7:
                        indicator = "🟡"
                    else:
                        indicator = "🔴"
                    
                    print(f"   {i}. {indicator} Relevance: {score:.3f}")
                    if metadata.get('source'):
                        print(f"      Source: {Path(metadata['source']).name}")
                    if metadata.get('page'):
                        print(f"      Page: {metadata['page']}")
                    print(f"      Preview: {content}")
            
            # Show persistence status
            if result.get("should_persist"):
                print(f"\n💾 Session will be persisted to long-term memory")
            
            # Show interaction count
            print(f"\n📊 Interaction: {self.interaction_count}")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
            print("   Check logs/cli.log for details")
    
    def run(self):
        """Run the CLI interface."""
        self.print_banner()
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = input("\n>>> ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Check if it's a command
                    if user_input.startswith('/'):
                        should_continue = self.process_command(user_input)
                        if not should_continue:
                            break
                    else:
                        # Process as query
                        self.process_query(user_input)
                
                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted. Use /exit to quit.")
                    continue
                
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            print(f"\n❌ Fatal error: {e}")
            print("   Check logs/cli.log for details")
        
        finally:
            print("\n" + "=" * 70)
            print("Session ended. Logs saved to logs/cli.log")
            print("=" * 70)


def main():
    """Main entry point."""
    try:
        cli = ResearchAssistantCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
