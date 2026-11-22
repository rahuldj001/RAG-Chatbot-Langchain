"""Chat Engine orchestration layer using LangChain.

Configures conversation mode based on whether documents have been uploaded:
1. `SimpleChatEngine`: Conversational memory buffer for chit-chat when no PDFs are uploaded.
2. `CondensePlusContextChatEngine`: RAG conversation engine that condenses multi-turn user
   questions into standalone queries, retrieves relevant document chunks, and answers factually.
"""

from typing import Any, Generator, List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from .retriever import LocalRetriever, ScoredDocument
from ..prompt.qa_prompt import (
    get_context_prompt,
    get_system_prompt,
    CONDENSED_CONTEXT_PROMPT_EN,
    CONDENSED_CONTEXT_PROMPT_VI,
)
from ...setting import RAGSettings

load_dotenv()


class StreamingChatResponse:
    """Encapsulates a streaming token generator for real-time conversational responses.

    Yields individual response tokens on-the-fly as they arrive from the LLM stream,
    powering typewriter-style live text generation in the Gradio web UI.

    Attributes:
        _tokens: Cache of all emitted tokens assembled during iteration.
        _generator: The underlying token generator from the LangChain chat model.
        source_nodes: List of ScoredDocument chunks retrieved and used as grounding context.
    """

    def __init__(self, token_generator: Generator[str, None, None], source_nodes: List[ScoredDocument] | None = None):
        self._tokens: List[str] = []
        self._generator = token_generator
        self.source_nodes = source_nodes or []

    @property
    def response_gen(self) -> Generator[str, None, None]:
        """Generator yielding tokens sequentially for real-time frontend streaming."""
        for token in self._generator:
            self._tokens.append(token)
            yield token

    @property
    def response(self) -> str:
        """Synchronously consumes the remaining generator tokens and returns the complete text."""
        if not self._tokens:
            for _ in self.response_gen:
                pass
        return "".join(self._tokens)

    def __str__(self) -> str:
        return self.response


class SimpleChatEngine:
    """General conversational chat engine with token-bounded memory buffer.

    Active when no PDF documents are uploaded. Operates as a direct multi-turn conversational
    assistant, appending user inputs and assistant replies into a rolling memory buffer up to
    `token_limit` (default: 4000 tokens).
    """

    def __init__(self, llm: Any, system_prompt: str = "", token_limit: int = 4000):
        self.llm = llm
        self.system_prompt = system_prompt
        self.token_limit = token_limit
        self.history: List[HumanMessage | AIMessage] = []

    def reset(self):
        """Clears conversational memory history."""
        self.history = []

    def stream_chat(
        self, message: str, chat_history: List[Any] | None = None
    ) -> StreamingChatResponse:
        """Assembles conversational history and streams the assistant reply."""
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        # Use passed chat history or internal history
        if chat_history:
            for msg in chat_history:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    if str(msg.role).lower() in ["user", "human"]:
                        messages.append(HumanMessage(content=msg.content))
                    else:
                        messages.append(AIMessage(content=msg.content))
                elif isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ["user", "human"]:
                        messages.append(HumanMessage(content=content))
                    else:
                        messages.append(AIMessage(content=content))
        else:
            messages.extend(self.history)

        messages.append(HumanMessage(content=message))

        def _stream() -> Generator[str, None, None]:
            accumulated = []
            try:
                if hasattr(self.llm, "stream"):
                    for chunk in self.llm.stream(messages):
                        token = chunk.content if hasattr(chunk, "content") else str(chunk)
                        accumulated.append(token)
                        yield token
                elif hasattr(self.llm, "invoke"):
                    res = self.llm.invoke(messages)
                    token = res.content if hasattr(res, "content") else str(res)
                    accumulated.append(token)
                    yield token
            finally:
                full_reply = "".join(accumulated)
                self.history.append(HumanMessage(content=message))
                self.history.append(AIMessage(content=full_reply))

        return StreamingChatResponse(_stream())

    def chat(self, message: str) -> str:
        """Synchronous chat interface returning complete answer string."""
        res = self.stream_chat(message)
        return res.response


class CondensePlusContextChatEngine:
    """Retrieval-Augmented Generation (RAG) conversational engine.

    Executes a three-phase question answering cycle:
    1. Query Condensation: If prior dialogue exists, invokes the LLM to rewrite the latest
       follow-up question into a standalone, context-independent search query (resolving pronouns
       like 'it', 'they', or 'the previous method').
    2. Context Retrieval: Passes the standalone query into the hybrid retriever to fetch the
       top grounding document chunks.
    3. Grounded Answer Generation: Injects retrieved chunks into a system context prompt,
       instructing the LLM to answer strictly based on the provided document passages.
    """

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        system_prompt: str = "",
        language: str = "eng",
        token_limit: int = 4000,
    ):
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt or get_system_prompt(language, is_rag_prompt=True)
        self.language = language
        self.token_limit = token_limit
        self.history: List[HumanMessage | AIMessage] = []

    def reset(self):
        self.history = []

    def _condense_question(self, question: str, history: List[Any]) -> str:
        """Condenses conversation history + latest question into a standalone query."""
        if not history:
            return question

        history_str = ""
        for msg in history:
            role = "User" if isinstance(msg, HumanMessage) or getattr(msg, "role", "") in ["user", "human"] else "AI"
            content = msg.content if hasattr(msg, "content") else str(msg)
            history_str += f"{role}: {content}\n"

        prompt_template = (
            CONDENSED_CONTEXT_PROMPT_VI
            if self.language == "vi"
            else CONDENSED_CONTEXT_PROMPT_EN
        )
        prompt = prompt_template.format(chat_history=history_str.strip(), question=question)

        try:
            if hasattr(self.llm, "invoke"):
                res = self.llm.invoke(prompt)
                condensed = res.content if hasattr(res, "content") else str(res)
            elif hasattr(self.llm, "complete"):
                condensed = str(self.llm.complete(prompt))
            else:
                condensed = str(self.llm(prompt))
            return condensed.strip() or question
        except Exception:
            return question

    def stream_chat(
        self, message: str, chat_history: List[Any] | None = None
    ) -> StreamingChatResponse:
        current_history: List[HumanMessage | AIMessage] = []
        if chat_history:
            for msg in chat_history:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    if str(msg.role).lower() in ["user", "human"]:
                        current_history.append(HumanMessage(content=msg.content))
                    else:
                        current_history.append(AIMessage(content=msg.content))
                elif isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ["user", "human"]:
                        current_history.append(HumanMessage(content=content))
                    else:
                        current_history.append(AIMessage(content=content))
        else:
            current_history.extend(self.history)

        # 1. Condense into standalone query if history is present
        standalone_query = self._condense_question(message, current_history)

        # 2. Retrieve relevant document chunks
        if hasattr(self.retriever, "retrieve"):
            retrieved_nodes = self.retriever.retrieve(standalone_query)
        elif hasattr(self.retriever, "invoke"):
            retrieved_nodes = self.retriever.invoke(standalone_query)
        else:
            retrieved_nodes = []

        # 3. Format retrieved context
        context_str = "\n\n".join(
            [f"Document [{getattr(n, 'node_id', idx)}]: {n.page_content}" for idx, n in enumerate(retrieved_nodes)]
        )
        context_prompt_tmpl = get_context_prompt(self.language)
        context_augmented_prompt = context_prompt_tmpl.format(context_str=context_str)

        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        for msg in current_history:
            messages.append(msg)
        messages.append(HumanMessage(content=f"{context_augmented_prompt}\n\nQuestion: {message}"))

        def _stream() -> Generator[str, None, None]:
            accumulated = []
            try:
                if hasattr(self.llm, "stream"):
                    for chunk in self.llm.stream(messages):
                        token = chunk.content if hasattr(chunk, "content") else str(chunk)
                        accumulated.append(token)
                        yield token
                elif hasattr(self.llm, "invoke"):
                    res = self.llm.invoke(messages)
                    token = res.content if hasattr(res, "content") else str(res)
                    accumulated.append(token)
                    yield token
            finally:
                full_reply = "".join(accumulated)
                self.history.append(HumanMessage(content=message))
                self.history.append(AIMessage(content=full_reply))

        return StreamingChatResponse(_stream(), source_nodes=retrieved_nodes)

    def chat(self, message: str) -> str:
        res = self.stream_chat(message)
        return res.response


class LocalChatEngine:
    def __init__(
        self, setting: RAGSettings | None = None, host: str = "host.docker.internal"
    ):
        self._setting = setting or RAGSettings()
        self._retriever = LocalRetriever(self._setting, host=host)
        self._host = host

    def set_engine(
        self,
        llm: Any,
        nodes: List[Document],
        language: str = "eng",
        embed_model: Any | None = None,
        system_prompt: str | None = None,
    ) -> CondensePlusContextChatEngine | SimpleChatEngine:
        """Constructs the appropriate chat engine depending on document availability."""
        # 1. Mode A: No documents uploaded -> Standard Conversational Chat
        if len(nodes) == 0:
            return SimpleChatEngine(
                llm=llm,
                system_prompt=system_prompt or get_system_prompt(language, is_rag_prompt=False),
                token_limit=self._setting.ollama.chat_token_limit,
            )

        # 2. Mode B: Documents ingested -> Full RAG Chat Engine
        retriever = self._retriever.get_retrievers(
            nodes=nodes, llm=llm, language=language, embed_model=embed_model
        )
        return CondensePlusContextChatEngine(
            retriever=retriever,
            llm=llm,
            system_prompt=system_prompt or get_system_prompt(language, is_rag_prompt=True),
            language=language,
            token_limit=self._setting.ollama.chat_token_limit,
        )
