# src/core/rag_system.py

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import tiktoken
from loguru import logger
import hashlib

@dataclass
class Document:
    """Document structure for RAG system."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata
        }


class RAGSystem:
    """Retrieval-Augmented Generation system for tourism knowledge."""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 milvus_host: str = "localhost",
                 milvus_port: int = 19530,
                 collection_name: str = "tourism_knowledge"):
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Connect to Milvus
        connections.connect("default", host=milvus_host, port=milvus_port)
        
        # Initialize collection
        self.collection_name = collection_name
        self.collection = self._init_milvus_collection()
        
        # Initialize BM25 for hybrid search
        self.bm25 = None
        self.documents = []
        
        # Initialize tokenizer for chunk splitting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"RAG System initialized with collection: {collection_name}")
        
    def _init_milvus_collection(self) -> Collection:
        """Initialize Milvus collection for vector storage."""
        if utility.has_collection(self.collection_name):
            return Collection(self.collection_name)
            
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        schema = CollectionSchema(fields, "Tourism knowledge base")
        collection = Collection(self.collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)
        
        return collection
    
    def chunk_text(self, text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
        """Split text into chunks based on token count."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk)
            
        return chunks
    
    def add_document(self, content: str, metadata: Dict[str, Any], doc_id: Optional[str] = None):
        """Add a document to the knowledge base."""
        # Generate ID if not provided
        if not doc_id:
            doc_id = hashlib.md5(content.encode()).hexdigest()[:16]
            
        # Chunk the content
        chunks = self.chunk_text(content)
        
        docs = []
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            
            # Create embedding
            embedding = self.embedding_model.encode(chunk)
            
            # Prepare document
            doc = Document(
                id=chunk_id,
                content=chunk,
                metadata={**metadata, 'chunk_index': i, 'parent_id': doc_id},
                embedding=embedding
            )
            
            docs.append(doc)
            embeddings.append(embedding)
            
        # Insert into Milvus
        self._insert_to_milvus(docs)
        
        # Update BM25 index
        self.documents.extend(docs)
        self._update_bm25_index()
        
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
        
    def _insert_to_milvus(self, documents: List[Document]):
        """Insert documents into Milvus."""
        if not documents:
            return
            
        data = [
            [doc.id for doc in documents],
            [doc.embedding.tolist() for doc in documents],
            [doc.content for doc in documents],
            [doc.metadata for doc in documents]
        ]
        
        self.collection.insert(data)
        self.collection.flush()
        
    def _update_bm25_index(self):
        """Update BM25 index with current documents."""
        if self.documents:
            tokenized_docs = [doc.content.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            
    def hybrid_search(self, 
                     query: str, 
                     top_k: int = 10,
                     vector_weight: float = 0.7,
                     keyword_weight: float = 0.3,
                     filters: Optional[Dict] = None) -> List[Document]:
        """Perform hybrid search combining vector and keyword search."""
        
        # Vector search
        vector_results = self._vector_search(query, top_k * 2, filters)
        
        # BM25 keyword search
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Combine and rerank
        combined_results = self._combine_results(
            vector_results, 
            keyword_results,
            vector_weight,
            keyword_weight
        )
        
        return combined_results[:top_k]
    
    def _vector_search(self, query: str, top_k: int, filters: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Perform vector similarity search."""
        # Create query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Build search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # Build filter expression if provided
        expr = ""
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f'metadata["{key}"] == "{value}"')
                else:
                    conditions.append(f'metadata["{key}"] == {value}')
            expr = " && ".join(conditions) if conditions else ""
        
        # Load collection
        self.collection.load()
        
        # Search
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr if expr else None,
            output_fields=["content", "metadata"]
        )
        
        # Convert to documents
        docs_with_scores = []
        for hits in results:
            for hit in hits:
                doc = Document(
                    id=hit.id,
                    content=hit.entity.get('content'),
                    metadata=hit.entity.get('metadata')
                )
                docs_with_scores.append((doc, hit.distance))
                
        return docs_with_scores
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Perform BM25 keyword search."""
        if not self.bm25 or not self.documents:
            return []
            
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))
                
        return results
    
    def _combine_results(self, 
                        vector_results: List[Tuple[Document, float]],
                        keyword_results: List[Tuple[Document, float]],
                        vector_weight: float,
                        keyword_weight: float) -> List[Document]:
        """Combine and rerank results from vector and keyword search."""
        
        # Normalize scores
        vector_scores = self._normalize_scores(vector_results, inverse=True)  # L2 distance
        keyword_scores = self._normalize_scores(keyword_results)
        
        # Combine scores
        combined_scores = {}
        
        for doc, score in vector_scores:
            combined_scores[doc.id] = vector_weight * score
            
        for doc, score in keyword_scores:
            if doc.id in combined_scores:
                combined_scores[doc.id] += keyword_weight * score
            else:
                combined_scores[doc.id] = keyword_weight * score
                
        # Sort by combined score
        sorted_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get documents
        doc_map = {}
        for doc, _ in vector_results + keyword_results:
            doc_map[doc.id] = doc
            
        return [doc_map[doc_id] for doc_id, _ in sorted_docs if doc_id in doc_map]
    
    def _normalize_scores(self, results: List[Tuple[Document, float]], inverse: bool = False) -> List[Tuple[Document, float]]:
        """Normalize scores to 0-1 range."""
        if not results:
            return []
            
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [(doc, 1.0) for doc, _ in results]
            
        normalized = []
        for doc, score in results:
            if inverse:
                norm_score = 1 - (score - min_score) / (max_score - min_score)
            else:
                norm_score = (score - min_score) / (max_score - min_score)
            normalized.append((doc, norm_score))
            
        return normalized
    
    def load_tourism_data(self, data_path: str):
        """Load tourism data from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            content = self._format_tourism_content(item)
            metadata = {
                'type': item.get('type', 'attraction'),
                'name': item.get('name'),
                'location': item.get('location'),
                'category': item.get('category'),
                'price_range': item.get('price_range'),
                'rating': item.get('rating'),
                'opening_hours': item.get('opening_hours')
            }
            
            self.add_document(content, metadata, item.get('id'))
            
        logger.info(f"Loaded {len(data)} tourism items")
        
    def _format_tourism_content(self, item: Dict) -> str:
        """Format tourism item for indexing."""
        parts = [
            f"Name: {item.get('name', 'Unknown')}",
            f"Type: {item.get('type', 'attraction')}",
            f"Category: {item.get('category', 'general')}",
            f"Description: {item.get('description', '')}",
            f"Location: {item.get('address', '')}",
            f"Rating: {item.get('rating', 'N/A')}/5",
            f"Price Range: {item.get('price_range', 'N/A')}",
            f"Opening Hours: {item.get('opening_hours', 'N/A')}",
            f"Features: {', '.join(item.get('features', []))}",
            f"Reviews: {item.get('review_summary', '')}"
        ]
        
        return "\n".join(parts)
    
    def get_context_for_llm(self, query: str, filters: Optional[Dict] = None) -> str:
        """Get formatted context for LLM from search results."""
        results = self.hybrid_search(query, top_k=5, filters=filters)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"[Source {i}]")
            context_parts.append(doc.content)
            context_parts.append("")
        
        return "\n".join(context_parts)
