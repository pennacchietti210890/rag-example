import logging
import uuid
from threading import Lock
from typing import Dict, List, Optional

from .rag.rag import DocumentManager

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages document sessions"""

    def __init__(self):
        self._sessions: Dict[str, DocumentManager] = {}
        self._lock = Lock()

    def create_session(self) -> str:
        """Create a new session and return its ID"""
        with self._lock:
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = DocumentManager()
            logger.info(
                f"Created new session: {session_id}. Total sessions: {len(self._sessions)}"
            )
            return session_id

    def get_session(self, session_id: str) -> Optional[DocumentManager]:
        """Get a session by ID"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                logger.info(
                    f"Found session {session_id}, initialized: {session.is_initialized}"
                )
            else:
                logger.warning(f"Session {session_id} not found")
            return session

    def remove_session(self, session_id: str):
        """Remove a session"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions.pop(session_id)
                logger.info(
                    f"Removed session {session_id}. Total sessions: {len(self._sessions)}"
                )
            else:
                logger.warning(f"Attempted to remove non-existent session {session_id}")

    def list_sessions(self) -> List[str]:
        """List all active session IDs"""
        with self._lock:
            return list(self._sessions.keys())
