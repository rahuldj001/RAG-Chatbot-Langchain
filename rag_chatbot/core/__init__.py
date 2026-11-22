"""
Core RAG package exposing foundational building blocks:
embeddings, LLM model wrappers, ingestion pipelines, vector storage, chat engines, and prompts.
"""

from .embedding import LocalEmbedding
from .model import LocalRAGModel
from .ingestion import LocalDataIngestion
from .vector_store import LocalVectorStore
from .engine import LocalChatEngine
from .prompt import get_system_prompt

__all__ = [
    "LocalEmbedding",
    "LocalRAGModel",
    "LocalDataIngestion",
    "LocalVectorStore",
    "LocalChatEngine",
    "get_system_prompt",
]

