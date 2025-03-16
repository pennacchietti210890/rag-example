import logging
from threading import Lock
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Raised when there's an error processing the uploaded document"""

    pass


class DocumentManager:
    """Manages document state and operations"""

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50, num_chunks: int = 3
    ):
        self._index: Optional[faiss.IndexFlatL2] = None
        self._chunks: List[str] = []
        self._lock = Lock()
        self._is_initialized = False
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_chunks = num_chunks
        self._original_text: Optional[str] = None  # Store the original text

    @property
    def is_initialized(self) -> bool:
        """Check if a document has been loaded and indexed"""
        return self._is_initialized

    @property
    def chunks(self) -> List[str]:
        """Get the current chunks"""
        return self._chunks.copy()

    def process_document(
        self, text: str, embedding_model: SentenceTransformer
    ) -> Dict[str, int]:
        """Process a document and create embeddings"""
        with self._lock:
            try:
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

                # Initialize FAISS index
                self._index = faiss.IndexFlatL2(embeddings.shape[1])
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
        self, embedding_model: SentenceTransformer
    ) -> Dict[str, int]:
        """Reprocess the document with current parameters"""
        if not self._original_text:
            raise DocumentProcessingError("No document to reprocess")
        return self.process_document(self._original_text, embedding_model)

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
            query_embedding = embedding_model.encode([query])
            D, I = self._index.search(np.array(query_embedding), k=k)
            return [self._chunks[i] for i in I[0]]

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
