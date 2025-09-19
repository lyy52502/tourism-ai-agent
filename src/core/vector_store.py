# src/core/vector_store.py

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import json
from datetime import datetime
import hashlib

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import chromadb
from chromadb.config import Settings
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from loguru import logger


@dataclass
class VectorDocument:
    """Standard document structure for vector stores."""
    id: str
    content: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata
        }


class VectorStoreBase(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[VectorDocument]):
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 10, filters: Optional[Dict] = None) -> List[VectorDocument]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete(self, doc_ids: List[str]):
        """Delete documents by ID."""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        pass
    
    @abstractmethod
    def update_document(self, document: VectorDocument):
        """Update an existing document."""
        pass


class MilvusVectorStore(VectorStoreBase):
    """Milvus implementation of vector store."""
    
    def __init__(self, 
                 collection_name: str,
                 embedding_dim: int,
                 host: str = "localhost",
                 port: int = 19530,
                 metric_type: str = "L2"):
        
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.metric_type = metric_type
        
        # Connect to Milvus
        connections.connect("default", host=host, port=port)
        
        # Initialize collection
        self.collection = self._init_collection()
        
        logger.info(f"Milvus vector store initialized: {collection_name}")
    
    def _init_collection(self) -> Collection:
        """Initialize or get Milvus collection."""
        
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            collection.load()
            return collection
        
        # Create new collection
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        schema = CollectionSchema(fields, f"Collection for {self.collection_name}")
        collection = Collection(self.collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": self.metric_type,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)
        collection.load()
        
        return collection
    
    def add_documents(self, documents: List[VectorDocument]):
        """Add documents to Milvus."""
        
        if not documents:
            return
        
        # Prepare data
        data = [
            [doc.id for doc in documents],
            [doc.embedding.tolist() if doc.embedding is not None else [] for doc in documents],
            [doc.content for doc in documents],
            [doc.metadata for doc in documents]
        ]
        
        # Insert
        self.collection.insert(data)
        self.collection.flush()
        
        logger.info(f"Added {len(documents)} documents to Milvus")
    
    def search(self, 
              query_embedding: np.ndarray, 
              top_k: int = 10,
              filters: Optional[Dict] = None) -> List[VectorDocument]:
        """Search for similar documents in Milvus."""
        
        # Build search parameters
        search_params = {
            "metric_type": self.metric_type,
            "params": {"nprobe": 10}
        }
        
        # Build filter expression
        expr = ""
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f'metadata["{key}"] == "{value}"')
                elif isinstance(value, (list, tuple)):
                    conditions.append(f'metadata["{key}"] in {list(value)}')
                else:
                    conditions.append(f'metadata["{key}"] == {value}')
            expr = " && ".join(conditions) if conditions else ""
        
        # Search
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr if expr else None,
            output_fields=["content", "metadata"]
        )
        
        # Convert to VectorDocument
        documents = []
        for hits in results:
            for hit in hits:
                doc = VectorDocument(
                    id=hit.id,
                    content=hit.entity.get('content'),
                    embedding=None,  # Don't return embeddings in search
                    metadata=hit.entity.get('metadata', {})
                )
                documents.append(doc)
        
        return documents
    
    def delete(self, doc_ids: List[str]):
        """Delete documents from Milvus."""
        
        expr = f'id in {doc_ids}'
        self.collection.delete(expr)
        self.collection.flush()
        
        logger.info(f"Deleted {len(doc_ids)} documents from Milvus")
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID from Milvus."""
        
        expr = f'id == "{doc_id}"'
        results = self.collection.query(
            expr=expr,
            output_fields=["content", "metadata"]
        )
        
        if results:
            return VectorDocument(
                id=doc_id,
                content=results[0].get('content'),
                embedding=None,
                metadata=results[0].get('metadata', {})
            )
        
        return None
    
    def update_document(self, document: VectorDocument):
        """Update a document in Milvus."""
        
        # Delete old version
        self.delete([document.id])
        
        # Add new version
        self.add_documents([document])


class ChromaVectorStore(VectorStoreBase):
    """ChromaDB implementation of vector store."""
    
    def __init__(self, 
                 collection_name: str,
                 persist_directory: Optional[str] = None):
        
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"ChromaDB vector store initialized: {collection_name}")
    
    def add_documents(self, documents: List[VectorDocument]):
        """Add documents to ChromaDB."""
        
        if not documents:
            return
        
        self.collection.add(
            ids=[doc.id for doc in documents],
            embeddings=[doc.embedding.tolist() if doc.embedding is not None else None for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents]
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(self, 
              query_embedding: np.ndarray,
              top_k: int = 10,
              filters: Optional[Dict] = None) -> List[VectorDocument]:
        """Search for similar documents in ChromaDB."""
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters
        )
        
        documents = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                doc = VectorDocument(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i] if 'documents' in results else '',
                    embedding=None,
                    metadata=results['metadatas'][0][i] if 'metadatas' in results else {}
                )
                documents.append(doc)
        
        return documents
    
    def delete(self, doc_ids: List[str]):
        """Delete documents from ChromaDB."""
        
        self.collection.delete(ids=doc_ids)
        logger.info(f"Deleted {len(doc_ids)} documents from ChromaDB")
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID from ChromaDB."""
        
        results = self.collection.get(ids=[doc_id])
        
        if results['ids']:
            return VectorDocument(
                id=doc_id,
                content=results['documents'][0] if results['documents'] else '',
                embedding=None,
                metadata=results['metadatas'][0] if results['metadatas'] else {}
            )
        
        return None
    
    def update_document(self, document: VectorDocument):
        """Update a document in ChromaDB."""
        
        self.collection.update(
            ids=[document.id],
            embeddings=[document.embedding.tolist()] if document.embedding is not None else None,
            documents=[document.content],
            metadatas=[document.metadata]
        )


class FAISSVectorStore(VectorStoreBase):
    """FAISS implementation of vector store."""
    
    def __init__(self, 
                 embedding_dim: int,
                 index_path: Optional[str] = None,
                 metric_type: str = "cosine"):
        
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metric_type = metric_type
        
        # Document storage (FAISS only stores vectors)
        self.documents = {}  # id -> document
        self.id_to_index = {}  # document id -> FAISS index
        self.index_to_id = {}  # FAISS index -> document id
        
        # Initialize FAISS index
        if metric_type == "cosine":
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine
        elif metric_type == "l2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Load existing index if path provided
        if index_path and os.path.exists(index_path):
            self.load(index_path)
        
        logger.info(f"FAISS vector store initialized with dim={embedding_dim}")
    
    def add_documents(self, documents: List[VectorDocument]):
        """Add documents to FAISS."""
        
        if not documents:
            return
        
        # Prepare embeddings
        embeddings = []
        for doc in documents:
            if doc.embedding is not None:
                # Normalize for cosine similarity
                if self.metric_type == "cosine":
                    norm = np.linalg.norm(doc.embedding)
                    if norm > 0:
                        embedding = doc.embedding / norm
                    else:
                        embedding = doc.embedding
                else:
                    embedding = doc.embedding
                
                embeddings.append(embedding)
                
                # Store document
                current_index = self.index.ntotal
                self.documents[doc.id] = doc
                self.id_to_index[doc.id] = current_index
                self.index_to_id[current_index] = doc.id
        
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
        
        logger.info(f"Added {len(documents)} documents to FAISS")
        
        # Save if path specified
        if self.index_path:
            self.save(self.index_path)
    
    def search(self, 
              query_embedding: np.ndarray,
              top_k: int = 10,
              filters: Optional[Dict] = None) -> List[VectorDocument]:
        """Search for similar documents in FAISS."""
        
        if self.index.ntotal == 0:
            return []
        
        # Normalize query for cosine similarity
        if self.metric_type == "cosine":
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
        
        # Search
        query_array = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
        
        # Get documents
        documents = []
        for idx in indices[0]:
            if idx >= 0 and idx in self.index_to_id:
                doc_id = self.index_to_id[idx]
                doc = self.documents.get(doc_id)
                
                if doc:
                    # Apply filters if provided
                    if filters:
                        match = True
                        for key, value in filters.items():
                            if key not in doc.metadata or doc.metadata[key] != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    documents.append(doc)
        
        return documents[:top_k]
    
    def delete(self, doc_ids: List[str]):
        """Delete documents from FAISS."""
        
        # FAISS doesn't support direct deletion, need to rebuild index
        remaining_docs = []
        
        for doc_id, doc in self.documents.items():
            if doc_id not in doc_ids:
                remaining_docs.append(doc)
        
        # Clear and rebuild
        self.documents.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()
        self.index = faiss.IndexFlatIP(self.embedding_dim) if self.metric_type == "cosine" else faiss.IndexFlatL2(self.embedding_dim)
        
        # Re-add remaining documents
        self.add_documents(remaining_docs)
        
        logger.info(f"Deleted {len(doc_ids)} documents from FAISS")
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID from FAISS."""
        return self.documents.get(doc_id)
    
    def update_document(self, document: VectorDocument):
        """Update a document in FAISS."""
        
        # Delete old version
        if document.id in self.documents:
            self.delete([document.id])
        
        # Add new version
        self.add_documents([document])
    
    def save(self, path: str):
        """Save FAISS index and documents to disk."""
        
        # Save index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save documents and mappings
        data = {
            'documents': {k: v.to_dict() for k, v in self.documents.items()},
            'id_to_index': self.id_to_index,
            'index_to_id': self.index_to_id
        }
        
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved FAISS index to {path}")
    
    def load(self, path: str):
        """Load FAISS index and documents from disk."""
        
        # Load index
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
        
        # Load documents and mappings
        if os.path.exists(f"{path}.pkl"):
            with open(f"{path}.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.documents = {
                k: VectorDocument(**v) for k, v in data['documents'].items()
            }
            self.id_to_index = data['id_to_index']
            self.index_to_id = data['index_to_id']
        
        logger.info(f"Loaded FAISS index from {path}")


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create(store_type: str, **kwargs) -> VectorStoreBase:
        """Create a vector store instance."""
        
        if store_type == "milvus":
            return MilvusVectorStore(**kwargs)
        elif store_type == "chroma":
            return ChromaVectorStore(**kwargs)
        elif store_type == "faiss":
            return FAISSVectorStore(**kwargs)
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> VectorStoreBase:
        """Create a vector store from configuration."""
        
        store_type = config.pop('type')
        return VectorStoreFactory.create(store_type, **config)