import logging
from threading import Lock
from typing import Dict, List, Optional, Tuple, Literal

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.rag.self_rag.self_rag_graph import self_rag_workflow
from langchain import hub

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Raised when there's an error processing the uploaded document"""

    pass


# Define supported distance metrics as a type
DistanceMetric = Literal["l2", "ip", "cosine", "hamming"]


class DocumentManager:
    """Manages document state and operations"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        num_chunks: int = 3,
        distance_metric: DistanceMetric = "l2",
    ):
        self._index: Optional[faiss.Index] = None
        self._chunks: List[str] = []
        self._lock = Lock()
        self._is_initialized = False
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_chunks = num_chunks
        self.distance_metric = distance_metric
        self._original_text: Optional[str] = None  # Store the original text

    @property
    def is_initialized(self) -> bool:
        """Check if a document has been loaded and indexed"""
        return self._is_initialized

    @property
    def chunks(self) -> List[str]:
        """Get the current chunks"""
        return self._chunks.copy()

    def _create_faiss_index(self, embedding_dim: int) -> faiss.Index:
        """Create a FAISS index based on the selected distance metric"""
        if self.distance_metric == "l2":
            # L2 distance (Euclidean)
            return faiss.IndexFlatL2(embedding_dim)
        elif self.distance_metric == "ip":
            # Inner product (dot product)
            return faiss.IndexFlatIP(embedding_dim)
        elif self.distance_metric == "cosine":
            # Cosine similarity (requires normalization)
            index = faiss.IndexFlatIP(embedding_dim)
            return index
        elif self.distance_metric == "hamming":
            # Hamming distance (for binary vectors)
            return faiss.IndexBinaryFlat(
                embedding_dim * 8
            )  # *8 because dimension is in bits for binary
        else:
            # Default to L2 if unknown metric specified
            logger.warning(
                f"Unknown distance metric '{self.distance_metric}', defaulting to L2"
            )
            return faiss.IndexFlatL2(embedding_dim)

    def process_document(
        self,
        text: str,
        embedding_model: SentenceTransformer,
        distance_metric: Optional[DistanceMetric] = None,
    ) -> Dict[str, int]:
        """Process a document and create embeddings"""
        with self._lock:
            try:
                # Update distance metric if provided
                if distance_metric:
                    self.distance_metric = distance_metric

                # Store the original text
                self._original_text = text

                # Split into chunks using instance variables
                self._chunks = self._chunk_text(text)

                if not self._chunks:
                    raise DocumentProcessingError(
                        "No valid chunks were created from the document"
                    )

                logger.info(f"Creating embeddings for {len(self._chunks)} chunks")
                embeddings = embedding_model.encode(self._chunks)

                # Initialize FAISS index based on distance metric
                self._index = self._create_faiss_index(embeddings.shape[1])

                # Normalize vectors if using cosine similarity
                if self.distance_metric == "cosine":
                    # Normalize vectors for cosine similarity
                    faiss.normalize_L2(embeddings)

                # Add embeddings to the index
                if self.distance_metric == "hamming":
                    # For hamming, convert to binary
                    binary_vectors = (embeddings > 0).astype(np.uint8)
                    self._index.add(binary_vectors)
                else:
                    # For other metrics, use as is
                    self._index.add(np.array(embeddings, dtype=np.float32))

                self._is_initialized = True
                return {"num_chunks": len(self._chunks)}

            except Exception as e:
                self._is_initialized = False
                self._index = None
                self._chunks = []
                self._original_text = None
                raise e

    def reprocess_document(
        self,
        embedding_model: SentenceTransformer,
        distance_metric: Optional[DistanceMetric] = None,
    ) -> Dict[str, int]:
        """Reprocess the document with current parameters"""
        if not self._original_text:
            raise DocumentProcessingError("No document to reprocess")

        # Update distance metric if provided
        if distance_metric:
            self.distance_metric = distance_metric

        return self.process_document(
            self._original_text, embedding_model, self.distance_metric
        )

    def search_chunks(
        self,
        query: str,
        embedding_model: SentenceTransformer,
        num_chunks: Optional[int] = None,
    ) -> List[str]:
        """Search for relevant chunks using the query"""
        with self._lock:
            if not self._is_initialized or self._index is None:
                raise DocumentProcessingError("No document has been loaded and indexed")

            # Use provided num_chunks or fall back to instance variable
            k = num_chunks if num_chunks is not None else self.num_chunks

            # Get query embedding
            query_embedding = embedding_model.encode([query])

            # Handle cosine similarity (normalize query vector)
            if self.distance_metric == "cosine":
                faiss.normalize_L2(query_embedding)

            # Handle hamming distance
            if self.distance_metric == "hamming":
                # Convert to binary for hamming distance
                query_binary = (query_embedding > 0).astype(np.uint8)
                D, I = self._index.search(query_binary, k=k)
            else:
                # Use normal search for other distance metrics
                D, I = self._index.search(np.array(query_embedding), k=k)

            return [self._chunks[i] for i in I[0]]

    def search_chunks_self_rag(
        self,
        query: str,
        embedding_model: SentenceTransformer,
        num_chunks: Optional[int] = None,
        model_name: str = "groq model not specified",
        api_key: str = "api key not available",
    ) -> Tuple[str, List[str]]:
        """Search for relevant chunks using the query and a self-rag pipeline"""
        with self._lock:
            if not self._is_initialized or self._index is None:
                raise DocumentProcessingError("No document has been loaded and indexed")

        initial_documents = self.search_chunks(query, embedding_model, num_chunks)
        workflow = self_rag_workflow()
        rag_prompt = hub.pull("rlm/rag-prompt")
        graph_response = workflow.invoke(
            {
                "question": query,
                "documents": initial_documents,
                "model_name": model_name,
                "api_key": api_key,
                "prompt": rag_prompt,
            }
        )
        return graph_response["generation"], graph_response["documents"]

    def reset(self):
        """Reset the document manager state"""
        with self._lock:
            self._index = None
            self._chunks = []
            self._is_initialized = False
            self._original_text = None

    def _chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        """Splits text into overlapping chunks for better retrieval"""
        if not text or not text.strip():
            raise DocumentProcessingError("Empty text provided for chunking")

        # Use provided parameters or fall back to instance variables
        size = chunk_size if chunk_size is not None else self.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap

        words = text.split()
        if not words:
            raise DocumentProcessingError("No words found in text after splitting")

        return [
            " ".join(words[i : i + size]) for i in range(0, len(words), size - overlap)
        ]
