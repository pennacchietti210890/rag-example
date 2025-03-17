import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from backend.rag.rag import DocumentManager, DocumentProcessingError


@pytest.mark.unit
class TestDocumentManager:
    
    def test_initialization(self):
        """Test that the DocumentManager initializes correctly"""
        # When
        manager = DocumentManager(chunk_size=200, chunk_overlap=50, num_chunks=3)
        
        # Then
        assert manager.chunk_size == 200
        assert manager.chunk_overlap == 50
        assert manager.num_chunks == 3
        assert manager.is_initialized is False
        assert manager.chunks == []
        assert manager._index is None
    
    def test_process_document(self, document_manager, sample_text, mock_embedding_model):
        """Test processing a document"""
        # When
        result = document_manager.process_document(sample_text, mock_embedding_model)
        
        # Then
        assert document_manager.is_initialized is True
        assert len(document_manager.chunks) > 0
        assert document_manager._index is not None
        assert result["num_chunks"] == len(document_manager.chunks)
        
        # Verify the mock was called
        mock_embedding_model.encode.assert_called()
    
    def test_process_empty_document(self, document_manager, mock_embedding_model):
        """Test processing an empty document raises an error"""
        # When/Then
        with pytest.raises(DocumentProcessingError):
            document_manager.process_document("", mock_embedding_model)
    
    def test_reprocess_document(self, document_manager, sample_text, mock_embedding_model):
        """Test reprocessing a document with new parameters"""
        # Given
        document_manager.process_document(sample_text, mock_embedding_model)
        original_chunks = document_manager.chunks.copy()
        
        # When - change parameters and reprocess
        document_manager.chunk_size = 50  # Smaller chunks
        result = document_manager.reprocess_document(mock_embedding_model)
        
        # Then
        assert document_manager.is_initialized is True
        assert len(document_manager.chunks) > 0
        # With smaller chunk size, we should have more chunks
        assert len(document_manager.chunks) > len(original_chunks)
        assert result["num_chunks"] == len(document_manager.chunks)
    
    def test_search_chunks(self, document_manager, sample_text, mock_embedding_model):
        """Test searching for relevant chunks"""
        # Given
        document_manager.process_document(sample_text, mock_embedding_model)
        
        # When
        query = "What was the revenue in Q1?"
        results = document_manager.search_chunks(query, mock_embedding_model)
        
        # Then
        assert len(results) > 0
        assert len(results) <= document_manager.num_chunks
        
        # Verify the mock was called with the query
        mock_embedding_model.encode.assert_called_with([query])
    
    def test_search_chunks_uninitialized(self, document_manager, mock_embedding_model):
        """Test searching when no document has been processed"""
        # When/Then
        with pytest.raises(DocumentProcessingError):
            document_manager.search_chunks("test query", mock_embedding_model)
    
    @patch('faiss.IndexFlatL2')
    def test_faiss_integration(self, mock_faiss_index, document_manager, sample_text, mock_embedding_model):
        """Test that FAISS is used correctly"""
        # Setup mock
        mock_index_instance = MagicMock()
        mock_faiss_index.return_value = mock_index_instance
        
        # When
        document_manager.process_document(sample_text, mock_embedding_model)
        
        # Then
        # Verify FAISS index was created and add was called
        mock_faiss_index.assert_called_once()
        assert mock_index_instance.add.call_count > 0 