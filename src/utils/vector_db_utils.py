"""Utility functions for vector database operations."""
import logging
from typing import List, Dict, Optional, Literal
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class VectorDB:
    """Base class for vector database operations."""
    
    def __init__(self, config: dict):
        """Initialize vector database.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def add_documents(
        self,
        documents: List[Dict],
        collection_name: str = "default"
    ):
        """Add documents to the database."""
        raise NotImplementedError
    
    def search(
        self,
        query_embedding: List[float],
        collection_name: str = "default",
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar documents."""
        raise NotImplementedError
    
    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        raise NotImplementedError


class ChromaDBClient(VectorDB):
    """ChromaDB vector database client."""
    
    def __init__(self, config: dict):
        """Initialize ChromaDB client.
        
        Args:
            config: Configuration dictionary with 'chromadb_path'
        """
        super().__init__(config)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            db_path = self.config.get("chromadb_path", "./data/chromadb")
            Path(db_path).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Initializing ChromaDB at {db_path}")
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info("ChromaDB initialized successfully")
            
        except ImportError:
            logger.error("chromadb not installed")
            raise ImportError("Please install chromadb (pip install chromadb)")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def add_documents(
        self,
        documents: List[Dict],
        collection_name: str = "default"
    ):
        """Add documents to ChromaDB.
        
        Args:
            documents: List of document dicts with 'text', 'embedding', and metadata
            collection_name: Name of the collection
        """
        if not self.client:
            raise RuntimeError("ChromaDB client not initialized")
        
        try:
            # Get or create collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Research Assistant documents"}
            )
            
            # Prepare data for insertion
            ids = []
            embeddings = []
            documents_text = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                doc_id = doc.get("id", f"doc_{i}_{hash(doc['text'][:100])}")
                ids.append(str(doc_id))
                embeddings.append(doc["embedding"])
                documents_text.append(doc["text"])
                
                # Extract metadata (exclude embedding and text)
                metadata = {k: v for k, v in doc.items() 
                           if k not in ["embedding", "text"]}
                # Convert non-string values to strings for ChromaDB
                metadata = {k: str(v) if not isinstance(v, (str, int, float, bool)) 
                           else v for k, v in metadata.items()}
                metadatas.append(metadata)
            
            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents_text,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        collection_name: str = "default",
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search ChromaDB for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            collection_name: Name of the collection
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of matching documents with scores
        """
        if not self.client:
            raise RuntimeError("ChromaDB client not initialized")
        
        try:
            # Get collection
            collection = self.client.get_or_create_collection(name=collection_name)
            
            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict
            )
            
            # Format results
            documents = []
            if results["documents"] and results["distances"] and len(results["documents"]) > 0:
                for i in range(len(results["documents"][0])):
                    doc = {
                        "text": results["documents"][0][i],
                        "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                    }
                    documents.append(doc)
            
            logger.info(f"Found {len(documents)} documents in collection '{collection_name}'")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str):
        """Delete a collection from ChromaDB."""
        if not self.client:
            raise RuntimeError("ChromaDB client not initialized")
        
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise


class PineconeClient(VectorDB):
    """Pinecone vector database client."""
    
    def __init__(self, config: dict):
        """Initialize Pinecone client.
        
        Args:
            config: Configuration dictionary with Pinecone settings
        """
        super().__init__(config)
        self.index = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Pinecone client."""
        try:
            import pinecone
            
            api_key = self.config.get("pinecone_api_key")
            if not api_key:
                raise ValueError("Pinecone API key not provided")
            
            environment = self.config.get("pinecone_environment", "us-east-1-aws")
            index_name = self.config.get("pinecone_index_name", "research-assistant")
            
            logger.info(f"Initializing Pinecone in {environment}")
            pinecone.init(api_key=api_key, environment=environment)
            
            # Create index if it doesn't exist
            if index_name not in pinecone.list_indexes():
                dimension = self.config.get("embedding_dimension", 384)
                pinecone.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine"
                )
                logger.info(f"Created Pinecone index '{index_name}'")
            
            self.index = pinecone.Index(index_name)
            logger.info("Pinecone initialized successfully")
            
        except ImportError:
            logger.error("pinecone-client not installed")
            raise ImportError("Please install pinecone-client (pip install pinecone-client)")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def add_documents(
        self,
        documents: List[Dict],
        collection_name: str = "default"
    ):
        """Add documents to Pinecone.
        
        Args:
            documents: List of document dicts with 'text', 'embedding', and metadata
            collection_name: Namespace in Pinecone
        """
        if not self.index:
            raise RuntimeError("Pinecone client not initialized")
        
        try:
            vectors = []
            for i, doc in enumerate(documents):
                doc_id = doc.get("id", f"doc_{i}_{hash(doc['text'][:100])}")
                
                # Prepare metadata (Pinecone has size limits)
                metadata = {k: v for k, v in doc.items() 
                           if k not in ["embedding", "text"]}
                metadata["text"] = doc["text"][:1000]  # Truncate text for metadata
                
                vectors.append((
                    str(doc_id),
                    doc["embedding"],
                    metadata
                ))
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=collection_name)
            
            logger.info(f"Added {len(documents)} documents to Pinecone namespace '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {str(e)}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        collection_name: str = "default",
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search Pinecone for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            collection_name: Namespace in Pinecone
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of matching documents with scores
        """
        if not self.index:
            raise RuntimeError("Pinecone client not initialized")
        
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=collection_name,
                filter=filter_dict,
                include_metadata=True
            )
            
            documents = []
            for match in results["matches"]:
                doc = {
                    "text": match["metadata"].get("text", ""),
                    "score": match["score"],
                    "metadata": match["metadata"]
                }
                documents.append(doc)
            
            logger.info(f"Found {len(documents)} documents in Pinecone namespace '{collection_name}'")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str):
        """Delete a namespace from Pinecone."""
        if not self.index:
            raise RuntimeError("Pinecone client not initialized")
        
        try:
            self.index.delete(delete_all=True, namespace=collection_name)
            logger.info(f"Deleted Pinecone namespace '{collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting namespace: {str(e)}")
            raise


def create_vector_db(
    db_type: Literal["chromadb", "pinecone"],
    config: dict
) -> VectorDB:
    """Factory function to create a vector database client.
    
    Args:
        db_type: Type of vector database
        config: Configuration dictionary
        
    Returns:
        VectorDB instance
    """
    if db_type == "chromadb":
        return ChromaDBClient(config)
    elif db_type == "pinecone":
        return PineconeClient(config)
    else:
        raise ValueError(f"Unsupported vector database type: {db_type}")
