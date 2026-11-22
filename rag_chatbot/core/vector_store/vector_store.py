"""Vector Store Index management module using LangChain.

Constructs in-memory vector store structures from document nodes for similarity search.
Reproduces the in-memory VectorStoreIndex behavior without requiring persistent storage.
"""

from typing import Any, List, Tuple
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from ...setting import RAGSettings

load_dotenv()


class InMemoryCosineVectorStore:
    """Lightweight in-memory vector store providing exact cosine similarity retrieval.

    Unlike external persistent databases (ChromaDB, Pinecone, FAISS), this store maintains
    document chunks and their dense embeddings directly in RAM as NumPy float32 arrays.
    
    Mathematical Foundations:
    - Normalizes document vectors to unit length: `D_norm = D / ||D||_2`.
    - Normalizes query vector to unit length: `q_norm = q / ||q||_2`.
    - Cosine similarity evaluates directly as a fast matrix dot product:
      `similarity = D_norm · q_norm`.
    - Sorts indices descending and returns the top `k` scoring documents.
    """

    def __init__(self, embedding_model: Any | None = None):
        self.embedding_model = embedding_model
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []

    def add_documents(self, documents: List[Document]):
        if not documents:
            return
        self.documents.extend(documents)
        # Check if embeddings are already present on documents
        missing_docs = []
        missing_indices = []
        for i, doc in enumerate(documents):
            emb = doc.metadata.get("embedding") or getattr(doc, "embedding", None)
            if emb is not None:
                self.embeddings.append(emb)
            else:
                missing_docs.append(doc.page_content)
                missing_indices.append(len(self.embeddings))
                self.embeddings.append([])

        if missing_docs and self.embedding_model:
            computed_embs = self.embedding_model.embed_documents(missing_docs)
            for idx, emb in zip(missing_indices, computed_embs):
                self.embeddings[idx] = emb
                self.documents[idx].metadata["embedding"] = emb

    def similarity_search_with_score(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        if not self.documents:
            return []
        query_embedding = (
            self.embedding_model.embed_query(query)
            if self.embedding_model
            else None
        )
        if query_embedding is None:
            return [(doc, 1.0) for doc in self.documents[:k]]

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        doc_matrix = np.array(self.embeddings, dtype=np.float32)
        # Normalize rows
        norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        doc_matrix = doc_matrix / norms

        scores = np.dot(doc_matrix, q_vec)
        top_indices = np.argsort(scores)[::-1][:k]

        return [(self.documents[i], float(scores[i])) for i in top_indices]

    def similarity_search(self, query: str, k: int = 20) -> List[Document]:
        results = self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in results]

    def as_retriever(self, **kwargs):
        class _Retriever:
            def __init__(self, store, search_kwargs):
                self.store = store
                self.k = search_kwargs.get("k", 20)

            def invoke(self, query: str) -> List[Document]:
                return self.store.similarity_search(query, k=self.k)

            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.invoke(query)

        search_kwargs = kwargs.get("search_kwargs", {})
        return _Retriever(self, search_kwargs)


class LocalVectorStore:
    def __init__(
        self,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None,
    ) -> None:
        self._setting = setting or RAGSettings()
        self._host = host

    def get_index(self, nodes: List[Document], embed_model: Any | None = None):
        """Builds and returns an in-memory vector index from a list of ingested text chunks."""
        if len(nodes) == 0:
            return None

        # Try to use LangChain's InMemoryVectorStore if available, otherwise custom cosine store
        try:
            from langchain_core.vectorstores import InMemoryVectorStore
            if embed_model is not None:
                store = InMemoryVectorStore(embedding=embed_model)
                store.add_documents(nodes)
                return store
        except Exception:
            pass

        store = InMemoryCosineVectorStore(embedding_model=embed_model)
        store.add_documents(nodes)
        return store
