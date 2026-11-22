"""Unit and Integration tests for LangChain RAG Chatbot modules."""

from typing import Any, List
import pytest
from langchain_core.documents import Document
from rag_chatbot.setting import RAGSettings
from rag_chatbot.core.ingestion.ingestion import LocalDataIngestion, SentenceSplitter, NodeDocument
from rag_chatbot.core.vector_store.vector_store import LocalVectorStore, InMemoryCosineVectorStore
from rag_chatbot.core.engine.retriever import (
    LangChainBM25Retriever,
    QueryFusionRetriever,
    TwoStageRetriever,
    RouterRetriever,
    ScoredDocument,
    dist_based_score_fusion,
)
from rag_chatbot.core.engine.engine import SimpleChatEngine, CondensePlusContextChatEngine
from rag_chatbot.core.prompt import get_system_prompt, get_context_prompt, get_query_gen_prompt


class DummyLLM:
    """Mock LLM for deterministic testing without external API calls."""

    def __init__(self, response_text: str = "Mock answer"):
        self.response_text = response_text

    def invoke(self, prompt: Any) -> Any:
        class _Msg:
            def __init__(self, c):
                self.content = c
        return _Msg(self.response_text)

    def stream(self, prompt: Any):
        for token in self.response_text.split():
            class _Msg:
                def __init__(self, c):
                    self.content = c + " "
            yield _Msg(token)

    def complete(self, prompt: str) -> str:
        return self.response_text


class DummyEmbeddings:
    """Mock Embeddings for testing vector search."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 10), 1.0, 0.5] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 10), 1.0, 0.5]


def test_text_filtering():
    """Validates regex-based PDF text sanitization and whitespace normalization."""
    ingestion = LocalDataIngestion()
    dirty_text = "Hello \x00 World!\n\nThis is a   test. \x08\x0bSpecial: Nguyễn Văn A."
    cleaned = ingestion._filter_text(dirty_text)
    assert "\x00" not in cleaned
    assert "\x08" not in cleaned
    assert "Hello World!" in cleaned
    assert "Nguyễn Văn A." in cleaned
    assert "   " not in cleaned


def test_sentence_splitter_chunking():
    """Validates SentenceSplitter reproduces sentence boundaries, chunk_size, and overlap."""
    splitter = SentenceSplitter(chunk_size=20, chunk_overlap=5)
    sample_text = (
        "First sentence here. Second sentence follows. Third sentence comes next.\n \n"
        "New paragraph with fourth sentence. Fifth sentence concludes."
    )
    doc = Document(page_content=sample_text, metadata={"file_name": "test.pdf"})
    nodes = splitter([doc])

    assert len(nodes) > 1
    for node in nodes:
        assert isinstance(node, NodeDocument)
        assert len(node.page_content) > 0
        assert node.metadata["file_name"] == "test.pdf"
        assert node.node_id is not None
        assert node.text == node.page_content


def test_node_document_compatibility():
    """Validates NodeDocument data structure, property accessors (.text, .node_id, .embedding), and content extraction."""
    node = NodeDocument(page_content="Sample passage text", metadata={"file_name": "doc.pdf"}, id="node-123")
    assert node.page_content == "Sample passage text"
    assert node.text == "Sample passage text"
    assert node.node_id == "node-123"
    assert node.get_content() == "Sample passage text"

    node.embedding = [0.1, 0.2, 0.3]
    assert node.embedding == [0.1, 0.2, 0.3]
    assert node.metadata["embedding"] == [0.1, 0.2, 0.3]


def test_distance_based_score_fusion():
    """Validates min-max distance-based score fusion with [0.4, 0.6] weights."""
    doc1 = Document(page_content="Doc 1", metadata={"node_id": "d1"})
    doc2 = Document(page_content="Doc 2", metadata={"node_id": "d2"})

    bm25_candidates = [(doc1, 10.0), (doc2, 5.0)]  # weight 0.4
    dense_candidates = [(doc2, 0.9), (doc1, 0.3)]  # weight 0.6

    fused = dist_based_score_fusion(
        [(bm25_candidates, 0.4), (dense_candidates, 0.6)],
        top_k=2,
    )

    assert len(fused) == 2
    assert isinstance(fused[0], ScoredDocument)
    assert fused[0].score >= fused[1].score


def test_in_memory_vector_store():
    """Validates InMemoryCosineVectorStore indexing and similarity search."""
    embedder = DummyEmbeddings()
    store = InMemoryCosineVectorStore(embedding_model=embedder)

    docs = [
        Document(page_content="Quantum computing uses qubits.", metadata={"node_id": "q1"}),
        Document(page_content="Machine learning optimizes algorithms.", metadata={"node_id": "ml1"}),
        Document(page_content="Deep neural networks learn representations.", metadata={"node_id": "nn1"}),
    ]
    store.add_documents(docs)

    results = store.similarity_search_with_score("qubits and computing", k=2)
    assert len(results) == 2
    assert results[0][1] >= results[1][1]


def test_bm25_retriever():
    """Validates BM25 keyword matching."""
    docs = [
        Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"node_id": "fox"}),
        Document(page_content="Artificial intelligence and neural networks.", metadata={"node_id": "ai"}),
    ]
    retriever = LangChainBM25Retriever(docs, k=2)
    results = retriever.retrieve("fox dog")
    assert len(results) > 0
    assert "fox" in results[0].page_content.lower()


def test_two_stage_retriever():
    """Validates TwoStageRetriever combines BM25 and Vector stores."""
    embedder = DummyEmbeddings()
    docs = [
        Document(page_content=f"Document chunk number {i} with unique information.", metadata={"node_id": f"d_{i}"})
        for i in range(10)
    ]
    v_store = InMemoryCosineVectorStore(embedding_model=embedder)
    v_store.add_documents(docs)
    bm25 = LangChainBM25Retriever(docs, k=10)

    # Use a dummy reranker by setting model directly
    retriever = TwoStageRetriever(
        retrievers=[bm25, v_store],
        retriever_weights=[0.4, 0.6],
        similarity_top_k=10,
        top_k_rerank=6,
    )
    class DummyRerankerModel:
        def predict(self, pairs):
            return [float(len(p[1])) for p in pairs]

    retriever.reranker.model = DummyRerankerModel()

    retrieved = retriever.retrieve("unique information")
    assert len(retrieved) <= 6
    assert isinstance(retrieved[0], ScoredDocument)


def test_simple_chat_engine_memory():
    """Validates SimpleChatEngine maintains conversational chat history."""
    llm = DummyLLM(response_text="Hello, how can I help you?")
    engine = SimpleChatEngine(llm=llm, system_prompt="System instructions")

    response = engine.stream_chat("Hi there!")
    reply = response.response
    assert "Hello" in reply
    assert len(engine.history) == 2

    # Second turn
    llm.response_text = "I remember our chat."
    response2 = engine.stream_chat("What did I say earlier?")
    reply2 = response2.response
    assert len(engine.history) == 4

    engine.reset()
    assert len(engine.history) == 0


def test_condense_plus_context_chat_engine():
    """Validates CondensePlusContextChatEngine question condensation and context retrieval."""
    docs = [Document(page_content="The capital of France is Paris.", metadata={"node_id": "paris"})]
    retriever = LangChainBM25Retriever(docs, k=1)
    llm = DummyLLM(response_text="The capital of France is Paris.")

    engine = CondensePlusContextChatEngine(
        retriever=retriever,
        llm=llm,
        language="eng",
    )

    response = engine.stream_chat("What is France's capital?")
    assert "Paris" in response.response
    assert len(response.source_nodes) > 0
    assert len(engine.history) == 2


def test_query_fusion_retriever():
    """Validates QueryFusionRetriever multi-query generation and distance fusion."""
    docs = [
        Document(page_content=f"Article {i} discussing machine learning architectures.", metadata={"node_id": f"art_{i}"})
        for i in range(5)
    ]
    bm25 = LangChainBM25Retriever(docs, k=5)
    v_store = InMemoryCosineVectorStore(embedding_model=DummyEmbeddings())
    v_store.add_documents(docs)

    llm = DummyLLM(response_text="1. ML architecture details\n2. Deep learning frameworks\n3. Neural network structures")
    fusion = QueryFusionRetriever(
        retrievers=[bm25, v_store],
        retriever_weights=[0.4, 0.6],
        llm=llm,
        query_gen_prompt=get_query_gen_prompt("eng"),
        similarity_top_k=3,
        num_queries=3,
    )

    results = fusion.retrieve("machine learning models")
    assert len(results) <= 3
    assert isinstance(results[0], ScoredDocument)


def test_router_retriever():
    """Validates RouterRetriever dynamically selects appropriate retriever based on query clarity."""
    docs = [Document(page_content=f"Knowledge base chunk {i}", metadata={"node_id": f"k_{i}"}) for i in range(3)]
    bm25 = LangChainBM25Retriever(docs, k=3)
    v_store = InMemoryCosineVectorStore(embedding_model=DummyEmbeddings())
    v_store.add_documents(docs)

    fusion = QueryFusionRetriever(
        retrievers=[bm25, v_store],
        retriever_weights=[0.4, 0.6],
        similarity_top_k=2,
    )
    two_stage = TwoStageRetriever(
        retrievers=[bm25, v_store],
        retriever_weights=[0.4, 0.6],
        similarity_top_k=2,
        top_k_rerank=2,
    )
    class DummyRerankerModel:
        def predict(self, pairs):
            return [1.0 for _ in pairs]
    two_stage.reranker.model = DummyRerankerModel()

    # LLM routes to choice 1
    router1 = RouterRetriever(fusion_retriever=fusion, two_stage_retriever=two_stage, llm=DummyLLM("1"), language="eng")
    res1 = router1.retrieve("ambiguous search term")
    assert len(res1) > 0

    # LLM routes to choice 2
    router2 = RouterRetriever(fusion_retriever=fusion, two_stage_retriever=two_stage, llm=DummyLLM("2"), language="eng")
    res2 = router2.retrieve("exact factual question")
    assert len(res2) > 0


def test_pipeline_initialization():
    """Validates LocalRAGPipeline initializes cleanly and toggles chat/QA modes."""
    from rag_chatbot.pipeline import LocalRAGPipeline

    pipeline = LocalRAGPipeline(host="localhost", default_model="openai/gpt-oss-20b")
    assert pipeline.get_language() == "eng"
    assert pipeline.get_model_name() == "openai/gpt-oss-20b"
    assert "artificial intelligence assistant" in pipeline.get_system_prompt()

    # Toggle language
    pipeline.set_language("vi")
    pipeline.set_chat_mode()
    assert pipeline.get_language() == "vi"
    assert "trợ lí trí tuệ nhân tạo" in pipeline.get_system_prompt()

    # Query in empty-document mode with mock LLM
    pipeline._default_model = DummyLLM("Greetings! How may I assist you?")
    pipeline.reset_engine()
    res = pipeline.query("chat", "Hello!", [])
    assert "Greetings" in res.response
