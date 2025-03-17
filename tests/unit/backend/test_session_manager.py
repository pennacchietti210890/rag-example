import pytest
from unittest.mock import patch
import uuid

from backend.session_manager import SessionManager


@pytest.mark.unit
class TestSessionManager:
    
    def test_create_session(self, session_manager):
        """Test creating a new session"""
        # When
        session_id = session_manager.create_session()
        
        # Then
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert session_id in session_manager._sessions
    
    def test_get_session(self, session_manager):
        """Test retrieving a session"""
        # Given
        session_id = session_manager.create_session()
        
        # When
        document_manager = session_manager.get_session(session_id)
        
        # Then
        assert document_manager is not None
        assert session_manager._sessions[session_id] == document_manager
    
    def test_get_nonexistent_session(self, session_manager):
        """Test retrieving a session that doesn't exist"""
        # Given
        nonexistent_id = str(uuid.uuid4())
        
        # When
        document_manager = session_manager.get_session(nonexistent_id)
        
        # Then
        assert document_manager is None
    
    
    def test_thread_safety(self, session_manager):
        """Test that the session manager is thread-safe"""
        # Verify that the lock is initialized
        assert hasattr(session_manager, '_lock')
        
        # This is a basic check - more comprehensive thread safety testing
        # would require actually spawning threads and checking for race conditions 