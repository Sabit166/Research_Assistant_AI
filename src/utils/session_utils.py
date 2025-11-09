"""Utility functions for session management."""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """Manage short-term session data and caching."""
    
    def __init__(self, cache_dir: str = "./data/sessions"):
        """Initialize session manager.
        
        Args:
            cache_dir: Directory for session cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Session manager initialized with cache dir: {cache_dir}")
    
    def save_session(
        self,
        session_id: str,
        session_data: Dict
    ):
        """Save session data to cache.
        
        Args:
            session_id: Unique session identifier
            session_data: Session data dictionary
        """
        try:
            session_file = self.cache_dir / f"{session_id}.json"
            
            # Add timestamp
            session_data["last_updated"] = datetime.now().isoformat()
            
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved session {session_id} to cache")
            
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {str(e)}")
            raise
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """Load session data from cache.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session data dictionary or None if not found
        """
        try:
            session_file = self.cache_dir / f"{session_id}.json"
            
            if not session_file.exists():
                logger.info(f"Session {session_id} not found in cache")
                return None
            
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
            
            logger.info(f"Loaded session {session_id} from cache")
            return session_data
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {str(e)}")
            return None
    
    def delete_session(self, session_id: str):
        """Delete session data from cache.
        
        Args:
            session_id: Unique session identifier
        """
        try:
            session_file = self.cache_dir / f"{session_id}.json"
            
            if session_file.exists():
                session_file.unlink()
                logger.info(f"Deleted session {session_id} from cache")
            else:
                logger.warning(f"Session {session_id} not found for deletion")
                
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {str(e)}")
            raise
    
    def list_sessions(self) -> List[str]:
        """List all cached session IDs.
        
        Returns:
            List of session IDs
        """
        try:
            session_files = self.cache_dir.glob("*.json")
            session_ids = [f.stem for f in session_files]
            logger.info(f"Found {len(session_ids)} cached sessions")
            return session_ids
            
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            return []
    
    def create_session_summary(
        self,
        session_id: str,
        messages: List[Dict],
        documents: List[Dict]
    ) -> str:
        """Create a text summary of a session for long-term storage.
        
        Args:
            session_id: Session identifier
            messages: List of conversation messages
            documents: List of session documents
            
        Returns:
            Text summary of the session
        """
        try:
            summary_parts = [
                f"Session ID: {session_id}",
                f"Timestamp: {datetime.now().isoformat()}",
                f"Number of messages: {len(messages)}",
                f"Number of documents: {len(documents)}",
                "\n--- Conversation Summary ---"
            ]
            
            # Summarize conversation
            for i, msg in enumerate(messages):
                role = msg.get("type", "unknown")
                content = msg.get("content", "")
                summary_parts.append(f"[{i+1}] {role}: {content[:200]}...")
            
            # Summarize documents
            if documents:
                summary_parts.append("\n--- Document Summary ---")
                doc_sources = set(doc.get("source", "unknown") for doc in documents)
                summary_parts.append(f"Sources: {', '.join(doc_sources)}")
                summary_parts.append(f"Total chunks: {len(documents)}")
            
            summary = "\n".join(summary_parts)
            logger.info(f"Created summary for session {session_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating session summary: {str(e)}")
            raise
