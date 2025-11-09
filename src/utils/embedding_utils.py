"""Utility functions for generating embeddings."""
import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded successfully")
            
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise ImportError(
                "Please install sentence-transformers "
                "(pip install sentence-transformers)"
            )
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise


def add_embeddings_to_chunks(
    chunks: List[Dict],
    embedding_generator: EmbeddingGenerator
) -> List[Dict]:
    """Add embeddings to chunk dictionaries.
    
    Args:
        chunks: List of chunk dictionaries
        embedding_generator: EmbeddingGenerator instance
        
    Returns:
        Chunks with embeddings added
    """
    logger.info(f"Adding embeddings to {len(chunks)} chunks")
    
    # Extract texts
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(texts)
    
    # Add embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    
    logger.info(f"Successfully added embeddings to all chunks")
    return chunks
