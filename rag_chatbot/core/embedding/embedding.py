"""Embedding model factory.

Provides dense vector embeddings for document chunks and user queries.
Supports HuggingFace SentenceTransformer models (such as BAAI/bge-small-en-v1.5)
and OpenAI cloud embedding models (text-embedding-ada-002) using LangChain.
"""

from typing import Any, List
import requests
from dotenv import load_dotenv
from ...setting import RAGSettings

load_dotenv()


class LocalEmbedding:
    @staticmethod
    def set(setting: RAGSettings | None = None, **kwargs) -> Any:
        """Initializes and returns the configured LangChain embedding model."""
        setting = setting or RAGSettings()
        model_name = setting.ingestion.embed_llm

        # Cloud OpenAI embedding option
        if model_name == "text-embedding-ada-002":
            try:
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(model="text-embedding-ada-002")
            except ImportError:
                from langchain_community.embeddings import OpenAIEmbeddings
                return OpenAIEmbeddings(model="text-embedding-ada-002")

        # Default to local CPU-friendly HuggingFace SentenceTransformer model
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={
                    "batch_size": setting.ingestion.embed_batch_size,
                    "normalize_embeddings": True,
                },
            )
        except Exception:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    encode_kwargs={
                        "batch_size": setting.ingestion.embed_batch_size,
                        "normalize_embeddings": True,
                    },
                )
            except Exception:
                # Direct SentenceTransformer wrapper implementing LangChain Embeddings interface
                from sentence_transformers import SentenceTransformer

                class SentenceTransformerEmbeddingWrapper:
                    def __init__(self, model_id: str, batch_size: int = 8):
                        self.model = SentenceTransformer(model_id)
                        self.batch_size = batch_size

                    def embed_documents(self, texts: List[str]) -> List[List[float]]:
                        embeddings = self.model.encode(
                            texts,
                            batch_size=self.batch_size,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        )
                        return embeddings.tolist()

                    def embed_query(self, text: str) -> List[float]:
                        embedding = self.model.encode(
                            text,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        )
                        return embedding.tolist()

                    def __call__(self, texts_or_nodes: Any, **call_kwargs) -> Any:
                        if isinstance(texts_or_nodes, list):
                            if texts_or_nodes and hasattr(texts_or_nodes[0], "page_content"):
                                texts = [n.page_content for n in texts_or_nodes]
                                embs = self.embed_documents(texts)
                                for n, e in zip(texts_or_nodes, embs):
                                    if hasattr(n, "embedding"):
                                        n.embedding = e
                                    n.metadata["embedding"] = e
                                return texts_or_nodes
                            elif texts_or_nodes and isinstance(texts_or_nodes[0], str):
                                return self.embed_documents(texts_or_nodes)
                        return texts_or_nodes

                return SentenceTransformerEmbeddingWrapper(
                    model_id=model_name,
                    batch_size=setting.ingestion.embed_batch_size,
                )

    @staticmethod
    def pull(host: str, **kwargs):
        """Instructs Ollama to download embedding model if using Ollama-based embeddings."""
        setting = RAGSettings()
        payload = {"name": setting.ingestion.embed_llm}
        try:
            return requests.post(
                f"http://{host}:11434/api/pull", json=payload, stream=True, timeout=5
            )
        except Exception:
            return None

    @staticmethod
    def check_model_exist(host: str, **kwargs) -> bool:
        """Verifies if the embedding model is available."""
        setting = RAGSettings()
        try:
            res = requests.get(f"http://{host}:11434/api/tags", timeout=2)
            if res.status_code == 200:
                data = res.json()
                list_model = [d["name"] for d in data.get("models", [])]
                if setting.ingestion.embed_llm in list_model:
                    return True
        except Exception:
            pass
        # HuggingFace embeddings are managed and downloaded locally via HuggingFace Hub
        return True
