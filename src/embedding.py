import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Handles text embeddings using sentence transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        if not texts:
            return np.array([]).reshape(0, -1)

        print(f"Generating embeddings for {len(texts)} texts...")

        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )

        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.model is None:
            return 0
        return self.model.get_sentence_embedding_dimension()

    def __call__(self, texts: List[str]):
        """Allow calling the instance directly"""
        return self.generate_embeddings(texts)
