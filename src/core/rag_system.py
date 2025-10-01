# src/core/rag_system.py

import os
import json
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import tiktoken
from loguru import logger

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }

class RAGSystem:
    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Chroma client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="db_new"
        ))
        self.collection_name = "tourism_knowledge"
        self.collection = self.client.get_or_create_collection("tourism_knowledge")

        # BM25 keyword search
        self.documents: List[Document] = []
        self.bm25 = None

        # Tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(f"RAGSystem initialized with Chroma collection '{self.collection_name}'")

    def chunk_text(self, text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i+max_tokens]
            chunk = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk)
        return chunks

    def add_document(self, content: str, metadata: Dict[str, Any], doc_id: Optional[str] = None):
        if not doc_id:
            doc_id = hashlib.md5(content.encode()).hexdigest()[:16]

        chunks = self.chunk_text(content)
        docs = []
        embeddings = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            embedding = self.embedding_model.encode(chunk)
            doc = Document(
                id=chunk_id,
                content=chunk,
                metadata={**metadata, "chunk_index": i, "parent_id": doc_id},
                embedding=embedding
            )
            docs.append(doc)
            embeddings.append(embedding)

        # Insert into Chroma
        self.collection.add(
            documents=[d.content for d in docs],
            metadatas=[d.metadata for d in docs],
            ids=[d.id for d in docs],
            embeddings=[d.embedding.tolist() for d in docs]
        )

        # Update BM25
        self.documents.extend(docs)
        self._update_bm25_index()
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")

    def _update_bm25_index(self):
        if self.documents:
            tokenized_docs = [doc.content.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)

    def hybrid_search(self, query: str, top_k: int = 10, vector_weight: float = 0.7, keyword_weight: float = 0.3, filters: Optional[Dict] = None) -> List[Document]:
        vector_results = self._vector_search(query, top_k*2, filters)
        keyword_results = self._keyword_search(query, top_k*2)
        combined = self._combine_results(vector_results, keyword_results, vector_weight, keyword_weight)
        return combined[:top_k]

    def _vector_search(self, query: str, top_k: int, filters: Optional[Dict] = None):
        query_embedding = self.embedding_model.encode(query)
        where = filters if filters else {}
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents","metadatas","ids"]
        )
        hits = []
        for doc_id, content, metadata in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
            hits.append((Document(id=doc_id, content=content, metadata=metadata), 1.0))  # score placeholder
        return hits

    def _keyword_search(self, query: str, top_k: int):
        if not self.bm25 or not self.documents:
            return []
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))
        return results

    def _combine_results(self, vector_results, keyword_results, vector_weight, keyword_weight):
        vector_scores = self._normalize_scores(vector_results)
        keyword_scores = self._normalize_scores(keyword_results)
        combined_scores = {}
        doc_map = {}
        for doc, score in vector_scores:
            combined_scores[doc.id] = vector_weight * score
            doc_map[doc.id] = doc
        for doc, score in keyword_scores:
            combined_scores[doc.id] = combined_scores.get(doc.id, 0) + keyword_weight * score
            doc_map[doc.id] = doc
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs]

    def _normalize_scores(self, results):
        if not results:
            return []
        scores = [score for _, score in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [(doc, 1.0) for doc, _ in results]
        return [(doc, (score - min_s)/(max_s - min_s)) for doc, score in results]

    def load_tourism_data(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            content = self._format_tourism_content(item)
            metadata = {
                "type": item.get("type","attraction"),
                "name": item.get("name"),
                "location": item.get("location"),
                "category": item.get("category"),
                "price_range": item.get("price_range"),
                "rating": item.get("rating"),
                "opening_hours": item.get("opening_hours")
            }
            self.add_document(content, metadata, item.get("id"))
        logger.info(f"Loaded {len(data)} tourism items")

    def _format_tourism_content(self, item: Dict) -> str:
        parts = [
            f"Name: {item.get('name','Unknown')}",
            f"Type: {item.get('type','attraction')}",
            f"Category: {item.get('category','general')}",
            f"Description: {item.get('description','')}",
            f"Location: {item.get('address','')}",
            f"Rating: {item.get('rating','N/A')}/5",
            f"Price Range: {item.get('price_range','N/A')}",
            f"Opening Hours: {item.get('opening_hours','N/A')}",
            f"Features: {', '.join(item.get('features',[]))}",
            f"Reviews: {item.get('review_summary','')}"
        ]
        return "\n".join(parts)

    def get_context_for_llm(self, query: str, filters: Optional[Dict] = None) -> str:
        results = self.hybrid_search(query, top_k=5, filters=filters)
        if not results:
            return "No relevant information found in the knowledge base."
        context = []
        for i, doc in enumerate(results, 1):
            context.append(f"[Source {i}]")
            context.append(doc.content)
            context.append("")
        return "\n".join(context)
