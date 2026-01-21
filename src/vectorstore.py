import os
import pickle
import time
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import faiss
from langchain_core.documents import Document
from src.embedding import EmbeddingManager
from src.observability import get_logger
from src.llm_observability import get_llm_wrapper


class FaissVectorStore:
    """FAISS-based vector store for document retrieval"""

    def __init__(
        self, store_path: str, embedding_manager: Optional[EmbeddingManager] = None
    ):
        """
        Initialize FAISS vector store

        Args:
            store_path: Path to store/load FAISS index
            embedding_manager: EmbeddingManager instance (creates default if None)
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.index = None
        self.documents = []
        self.metadata = []
        self.logger = get_logger()
        self.llm_wrapper = get_llm_wrapper()
        self.obs_manager = self.llm_wrapper.obs_manager

    def build_from_documents(self, documents: List[Document]):
        """
        Build FAISS index from documents

        Args:
            documents: List of LangChain Document objects
        """
        if not documents:
            print("No documents to build index from")
            return

        print(f"Building FAISS index from {len(documents)} documents...")

        # Extract text and metadata
        texts = [doc.page_content for doc in documents]
        self.documents = documents
        self.metadata = [doc.metadata for doc in documents]

        # Generate embeddings
        embeddings = self.embedding_manager.generate_embeddings(texts)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        self.index.add(normalized_embeddings)

        print(
            f"FAISS index built with {len(embeddings)} documents, dimension: {dimension}"
        )

    def save(self):
        """Save FAISS index and metadata to disk"""
        if self.index is None:
            print("No index to save")
            return

        # Save FAISS index
        index_path = self.store_path / "faiss.index"
        faiss.write_index(self.index, str(index_path))

        # Save documents and metadata
        data_path = self.store_path / "documents.pkl"
        with open(data_path, "wb") as f:
            pickle.dump({"documents": self.documents, "metadata": self.metadata}, f)

        print(f"Saved FAISS index and documents to {self.store_path}")

    def load(self):
        """Load FAISS index and metadata from disk"""
        index_path = self.store_path / "faiss.index"
        data_path = self.store_path / "documents.pkl"

        if not index_path.exists() or not data_path.exists():
            print(f"No existing store found at {self.store_path}")
            return

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))

            # Load documents and metadata
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]

            print(f"Loaded FAISS index with {len(self.documents)} documents")

        except Exception as e:
            print(f"Error loading store: {e}")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query for similar documents

        Args:
            query_text: Query text
            top_k: Number of top results to return

        Returns:
            List of dictionaries with document content, metadata, and similarity scores
        """
        if self.index is None:
            self.logger.warning("Query attempted with no index available")
            return []

        # Add observability to query operation
        with self.llm_wrapper.observe_vector_store_operation(
            "query", top_k=top_k, query_length=len(query_text)
        ) as ctx:
            # Generate query embedding
            embedding_start = time.time()
            query_embedding = self.embedding_manager.generate_embeddings([query_text])[
                0
            ]
            embedding_time = time.time() - embedding_start

            self.logger.debug(
                "Query embedding generated",
                query_length=len(query_text),
                embedding_time_seconds=embedding_time,
            )

            # Normalize for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.reshape(1, -1)

            # Search
            search_start = time.time()
            similarities, indices = self.index.search(
                query_embedding, min(top_k, len(self.documents))
            )
            search_time = time.time() - search_start

            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0 and idx < len(self.documents):
                    results.append(
                        {
                            "content": self.documents[idx].page_content,
                            "metadata": self.metadata[idx],
                            "similarity_score": float(similarity),
                            "rank": i + 1,
                            "id": f"doc_{idx}",
                        }
                    )

            ctx["results_count"] = len(results)
            ctx["embedding_time_seconds"] = embedding_time
            ctx["search_time_seconds"] = search_time

            self.logger.debug(
                "Vector store query completed",
                top_k=top_k,
                results_returned=len(results),
                embedding_time_seconds=embedding_time,
                search_time_seconds=search_time,
                avg_similarity=np.mean(similarities[0]) if similarities.any() else 0,
            )

            return results

    def __len__(self):
        """Return number of documents in store"""
        return len(self.documents)
