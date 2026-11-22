"""Core RAG Pipeline coordinator using LangChain.

Encapsulates all major components of the application:
1. LLM Model Management (LocalRAGModel)
2. Document Ingestion & Chunking (LocalDataIngestion)
3. Embedding Models (LocalEmbedding)
4. Vector Storage (LocalVectorStore)
5. Conversational Query Engines (LocalChatEngine)
"""

from typing import Any, List
from langchain_core.messages import HumanMessage, AIMessage
from .core import (
    LocalChatEngine,
    LocalDataIngestion,
    LocalRAGModel,
    LocalEmbedding,
    LocalVectorStore,
    get_system_prompt,
)
from .core.engine.engine import StreamingChatResponse
from .setting import RAGSettings


class LocalRAGPipeline:
    def __init__(
        self, host: str = "host.docker.internal", default_model: str = "openai/gpt-oss-20b"
    ) -> None:
        self._host = host
        self._language = "eng"
        self._model_name = default_model

        # Default system prompt for non-RAG general chat
        self._system_prompt = get_system_prompt("eng", is_rag_prompt=False)

        # Initialize chat engine builder
        self._engine = LocalChatEngine(host=host)

        # Initialize LLM instance (e.g. Groq, OpenAI, or Ollama)
        self._default_model = LocalRAGModel.set(self._model_name, host=host)

        # Active chat/query engine (holds state and memory)
        self._query_engine = None

        # Handles PDF parsing and chunk storage
        self._ingestion = LocalDataIngestion()

        # Vector store interface
        self._vector_store = LocalVectorStore(host=host)

        # Initialize embedding model
        self._embed_model = LocalEmbedding.set(host=host)

        # Start with default empty-document engine ready for conversation
        self.reset_engine()

    def get_model_name(self) -> str:
        """Returns the currently active LLM model identifier."""
        return self._model_name

    def set_model_name(self, model_name: str):
        """Sets the active model identifier."""
        self._model_name = model_name

    def get_language(self) -> str:
        """Returns current UI language ('eng' or 'vi')."""
        return self._language

    def set_language(self, language: str):
        """Updates the conversation language."""
        self._language = language

    def get_system_prompt(self) -> str:
        """Returns the current system prompt string."""
        return self._system_prompt

    def set_system_prompt(self, system_prompt: str | None = None):
        """Updates system prompt, automatically selecting RAG vs Chit-Chat prompt if none provided."""
        self._system_prompt = system_prompt or get_system_prompt(
            language=self._language, is_rag_prompt=self._ingestion.check_nodes_exist()
        )

    def set_model(self):
        """Re-initializes the LLM instance with the current model name and system prompt."""
        self._default_model = LocalRAGModel.set(
            model_name=self._model_name,
            system_prompt=self._system_prompt,
            host=self._host,
        )

    def reset_engine(self):
        """Re-builds a clean conversation engine without attached documents."""
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            nodes=[],
            language=self._language,
            embed_model=self._embed_model,
            system_prompt=self._system_prompt,
        )

    def reset_documents(self):
        """Clears all ingested PDF chunks from memory."""
        self._ingestion.reset()

    def clear_conversation(self):
        """Resets conversation chat history memory while retaining loaded documents."""
        if self._query_engine:
            self._query_engine.reset()

    def reset_conversation(self):
        """Full reset: clears chat history, resets engine, and restores default prompt."""
        self.reset_engine()
        self.set_system_prompt(
            get_system_prompt(language=self._language, is_rag_prompt=False)
        )

    def set_embed_model(self, model_name: str):
        """Switches the active embedding model."""
        setting = RAGSettings()
        setting.ingestion.embed_llm = model_name
        self._embed_model = LocalEmbedding.set(setting=setting, host=self._host)

    def pull_model(self, model_name: str):
        """Triggers model download on Ollama (no-op for API models)."""
        return LocalRAGModel.pull(self._host, model_name)

    def pull_embed_model(self, model_name: str):
        """Triggers embedding model download on Ollama."""
        return LocalEmbedding.pull(self._host, model_name)

    def check_exist(self, model_name: str) -> bool:
        """Verifies if the specified model is ready to use."""
        return LocalRAGModel.check_model_exist(self._host, model_name)

    def check_exist_embed(self, model_name: str) -> bool:
        """Verifies if the specified embedding model is ready."""
        return LocalEmbedding.check_model_exist(self._host, model_name)

    def store_nodes(self, input_files: list[str] = None) -> None:
        """Parses, chunks, and indexes uploaded PDF documents into nodes."""
        self._ingestion.store_nodes(
            input_files=input_files or [],
            embed_nodes=True,
            embed_model=self._embed_model,
        )

    def set_chat_mode(self, system_prompt: str | None = None):
        """Re-configures the pipeline after document uploads or prompt/language updates."""
        self.set_language(self._language)
        self.set_system_prompt(system_prompt)
        self.set_model()
        self.set_engine()

    def set_engine(self):
        """Re-creates the query engine bound to all currently ingested document chunks."""
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            nodes=self._ingestion.get_ingested_nodes(),
            language=self._language,
            embed_model=self._embed_model,
            system_prompt=self._system_prompt,
        )

    def get_history(self, chatbot: list[list[str]]) -> List[HumanMessage | AIMessage]:
        """Converts Gradio [[user_msg, bot_msg], ...] tuples into LangChain Message objects."""
        history = []
        for chat in chatbot:
            if chat and len(chat) >= 2:
                if chat[0]:
                    history.append(HumanMessage(content=chat[0]))
                if chat[1]:
                    history.append(AIMessage(content=chat[1]))
        return history

    def query(
        self, mode: str, message: str, chatbot: list[list[str]]
    ) -> StreamingChatResponse:
        """Routes query to the engine and returns a token generator stream for real-time UI typing.

        - 'chat' mode: maintains multi-turn conversational context history.
        - 'QA' mode: single-turn independent question answering.
        """
        if mode == "chat":
            history = self.get_history(chatbot)
            return self._query_engine.stream_chat(message, history)
        else:
            self._query_engine.reset()
            return self._query_engine.stream_chat(message)
