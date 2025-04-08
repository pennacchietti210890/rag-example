import pytest
import numpy as np
import faiss
from unittest.mock import patch, MagicMock

from backend.rag.rag import DocumentManager, DocumentProcessingError


@pytest.mark.unit
class TestDocumentManager:
    def test_initialization(self):
        """Test that the DocumentManager initializes correctly"""
        # When
        manager = DocumentManager(
            chunk_size=200, chunk_overlap=50, num_chunks=3, distance_metric="l2"
        )

        # Then
        assert manager.chunk_size == 200
        assert manager.chunk_overlap == 50
        assert manager.num_chunks == 3
        assert manager.distance_metric == "l2"
        assert manager.is_initialized is False
        assert manager.chunks == []
        assert manager._index is None

    def test_initialization_with_different_distance_metric(self):
        """Test that the DocumentManager initializes correctly with different distance metrics"""
        # When
        metrics = ["l2", "ip", "cosine", "hamming"]
        for metric in metrics:
            manager = DocumentManager(
                chunk_size=200, chunk_overlap=50, num_chunks=3, distance_metric=metric
            )

            # Then
            assert manager.distance_metric == metric

    def test_process_document(
        self, document_manager, sample_text, mock_embedding_model
    ):
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

    def test_process_document_with_distance_metric(
        self, document_manager, sample_text, mock_embedding_model
    ):
        """Test processing a document with a specific distance metric"""
        # When
        result = document_manager.process_document(
            sample_text, mock_embedding_model, "ip"
        )

        # Then
        assert document_manager.is_initialized is True
        assert document_manager.distance_metric == "ip"
        assert len(document_manager.chunks) > 0
        assert document_manager._index is not None
        assert result["num_chunks"] == len(document_manager.chunks)

    def test_process_empty_document(self, document_manager, mock_embedding_model):
        """Test processing an empty document raises an error"""
        # When/Then
        with pytest.raises(DocumentProcessingError):
            document_manager.process_document("", mock_embedding_model)

    def test_reprocess_document(
        self, document_manager, sample_text, mock_embedding_model
    ):
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

    def test_reprocess_document_with_new_distance_metric(
        self, document_manager, sample_text, mock_embedding_model
    ):
        """Test reprocessing a document with a new distance metric"""
        # Given
        document_manager.process_document(sample_text, mock_embedding_model)
        assert document_manager.distance_metric == "l2"  # Default

        # When - change distance metric and reprocess
        result = document_manager.reprocess_document(mock_embedding_model, "ip")

        # Then
        assert document_manager.is_initialized is True
        assert document_manager.distance_metric == "ip"
        assert len(document_manager.chunks) > 0
        assert document_manager._index is not None
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

    @patch("faiss.IndexFlatL2")
    def test_create_faiss_index_l2(self, mock_index_flat_l2, document_manager):
        """Test that the correct FAISS index is created for L2 distance"""
        # Setup
        document_manager.distance_metric = "l2"
        embedding_dim = 128

        # Mock the returned index
        mock_index = MagicMock()
        mock_index_flat_l2.return_value = mock_index

        # When
        index = document_manager._create_faiss_index(embedding_dim)

        # Then
        mock_index_flat_l2.assert_called_once_with(embedding_dim)
        assert index == mock_index

    @patch("faiss.IndexFlatIP")
    def test_create_faiss_index_ip(self, mock_index_flat_ip, document_manager):
        """Test that the correct FAISS index is created for inner product"""
        # Setup
        document_manager.distance_metric = "ip"
        embedding_dim = 128

        # Mock the returned index
        mock_index = MagicMock()
        mock_index_flat_ip.return_value = mock_index

        # When
        index = document_manager._create_faiss_index(embedding_dim)

        # Then
        mock_index_flat_ip.assert_called_once_with(embedding_dim)
        assert index == mock_index

    @patch("faiss.IndexFlatIP")
    def test_create_faiss_index_cosine(self, mock_index_flat_ip, document_manager):
        """Test that the correct FAISS index is created for cosine similarity"""
        # Setup
        document_manager.distance_metric = "cosine"
        embedding_dim = 128

        # Mock the returned index
        mock_index = MagicMock()
        mock_index_flat_ip.return_value = mock_index

        # When
        index = document_manager._create_faiss_index(embedding_dim)

        # Then
        mock_index_flat_ip.assert_called_once_with(embedding_dim)
        assert index == mock_index

    @patch("faiss.IndexBinaryFlat")
    def test_create_faiss_index_hamming(self, mock_index_binary_flat, document_manager):
        """Test that the correct FAISS index is created for hamming distance"""
        # Setup
        document_manager.distance_metric = "hamming"
        embedding_dim = 128

        # Mock the returned index
        mock_index = MagicMock()
        mock_index_binary_flat.return_value = mock_index

        # When
        index = document_manager._create_faiss_index(embedding_dim)

        # Then
        mock_index_binary_flat.assert_called_once_with(
            embedding_dim * 8
        )  # *8 because dimension is in bits for binary
        assert index == mock_index

    @patch("faiss.normalize_L2")
    @patch("faiss.IndexFlatIP")
    def test_process_document_normalizes_for_cosine(
        self,
        mock_index_flat_ip,
        mock_normalize,
        document_manager,
        sample_text,
        mock_embedding_model,
    ):
        """Test that vectors are normalized when using cosine similarity"""
        # Setup
        document_manager.distance_metric = "cosine"
        mock_index = MagicMock()
        mock_index_flat_ip.return_value = mock_index

        # When
        document_manager.process_document(sample_text, mock_embedding_model)

        # Then
        # Verify normalize_L2 was called at least once (for embeddings)
        assert mock_normalize.call_count >= 1
        assert mock_index.add.call_count > 0
