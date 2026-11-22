"""Advanced Hybrid and Multi-Stage Retriever implementation using LangChain.

Combines:
1. Dense Vector Retrieval (semantic search via in-memory vector store)
2. Sparse Keyword Retrieval (exact keyword search via BM25Retriever)
3. Distance-based Score Fusion (merging candidate lists with weights [0.4, 0.6])
4. Query Generation (LLM rewriting user queries for multi-angle retrieval)
5. Cross-Encoder Reranking (scoring passage relevance via BAAI/bge-reranker-base)
6. Dynamic Router Retrieval (automatically routing queries based on clarity)
"""

import re
import numpy as np
from typing import Any, List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from ..ingestion.ingestion import NodeDocument
from ..vector_store.vector_store import LocalVectorStore
from ..prompt import get_query_gen_prompt, get_single_select_prompt
from ...setting import RAGSettings

load_dotenv()


class ScoredDocument(NodeDocument):
    """Document chunk paired with its computed retrieval or reranking relevance score.

    Throughout the multi-stage retrieval pipeline, documents receive numerical relevance
    scores from various estimators (cosine similarity, BM25 term frequency, or cross-encoder logits).
    This class wraps each chunk with its `.score` attribute while exposing unified properties:
    - `.page_content` / `.text`: The textual passage of the chunk.
    - `.score`: Floating-point relevance score (higher is more relevant).
    - `.node_id`: Unique chunk identifier.
    - `.metadata`: Document origin metadata (e.g. filename, page number).
    """
    score: float = 0.0

    def __init__(self, document: Document, score: float = 0.0, **kwargs):
        meta = dict(document.metadata)
        meta["score"] = score
        doc_id = getattr(document, "id", None) or meta.get("node_id")
        super().__init__(page_content=document.page_content, metadata=meta, id=doc_id, score=score, **kwargs)
        self.score = score

    @property
    def node(self):
        """Returns self as the underlying document chunk reference."""
        return self


class CrossEncoderReranker:
    """Neural cross-encoder reranker utilizing SentenceTransformers.

    Unlike bi-encoders (which encode queries and passages into separate vectors), a cross-encoder
    processes `[query, passage]` pairs simultaneously through full cross-attention layers. This allows
    all tokens in the query to attend directly to all tokens in the document, capturing nuanced semantic
    relationships, negations, and exact keyword alignments.

    Architecture & Lifecycle:
    - Default model: `BAAI/bge-reranker-base`.
    - Employs lazy loading: model weights are cached and downloaded only upon first inference,
      preventing startup latency or test suite stalls.
    - Input: Query string and a candidate list of `N` retrieved documents.
    - Output: Top `top_n` (default: 6) documents reordered descending by neural score.
    """

    _model_cache: dict[str, Any] = {}

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", top_n: int = 6):
        self.model_name = model_name
        self.top_n = top_n
        self._model = None

    @property
    def model(self):
        """Lazy-loaded CrossEncoder instance shared via class-level cache."""
        if self._model is not None:
            return self._model
        if self.model_name not in CrossEncoderReranker._model_cache:
            from sentence_transformers import CrossEncoder
            CrossEncoderReranker._model_cache[self.model_name] = CrossEncoder(self.model_name)
        return CrossEncoderReranker._model_cache[self.model_name]

    @model.setter
    def model(self, value: Any):
        self._model = value

    def rerank(self, query: str, documents: List[Document]) -> List[ScoredDocument]:
        """Scores all candidate documents against the query and returns top_n reranked results."""
        if not documents:
            return []
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        scored_docs = [
            ScoredDocument(doc, float(score))
            for doc, score in zip(documents, scores)
        ]
        # Sort descending by cross-encoder score
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        return scored_docs[: self.top_n]


def dist_based_score_fusion(
    candidate_lists_with_weights: List[Tuple[List[Tuple[Document, float]], float]],
    top_k: int = 6,
) -> List[ScoredDocument]:
    """Distance-Based Score Fusion combining candidate documents from heterogeneous retrievers.

    Different retrievers output scores on drastically different scales (e.g. BM25 scores are unbounded
    positive floats, whereas cosine similarity ranges from -1.0 to 1.0). To combine them fairly:

    Algorithm:
    1. Iterate over each retriever's candidate list:
       - Find the minimum (`min_s`) and maximum (`max_s`) score in that specific list.
       - Normalize each score to a [0.0, 1.0] range via Min-Max scaling:
         `norm_score = (raw_score - min_s) / (max_s - min_s)`
    2. Multiply the normalized score by the retriever's assigned weight:
       `weighted_score = weight * norm_score` (e.g., BM25 weight: 0.4, Dense weight: 0.6).
    3. Aggregate weighted scores across all lists by unique document identifier.
    4. Sort documents descending by total accumulated score and return top `top_k` results.
    """
    doc_scores: dict[str, float] = {}
    doc_store: dict[str, Document] = {}

    for candidates, weight in candidate_lists_with_weights:
        if not candidates:
            continue

        raw_scores = [score for _, score in candidates]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        denom = max_s - min_s if max_s > min_s else 1.0

        for doc, raw_score in candidates:
            # Generate consistent document key based on id or page content hash
            doc_id = getattr(doc, "id", None) or doc.metadata.get("node_id") or doc.page_content[:100]
            if max_s > min_s:
                norm_score = (raw_score - min_s) / denom
            else:
                norm_score = 1.0

            weighted_score = weight * norm_score
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + weighted_score
            if doc_id not in doc_store:
                doc_store[doc_id] = doc

    # Sort documents descending by total accumulated score
    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda did: doc_scores[did], reverse=True)
    fused_docs = [
        ScoredDocument(doc_store[did], doc_scores[did])
        for did in sorted_doc_ids[:top_k]
    ]
    return fused_docs


class LangChainBM25Retriever:
    """Sparse lexical keyword retriever wrapping the BM25Okapi algorithm.

    While dense vector models excel at semantic concept matching, they can struggle with
    exact part numbers, domain-specific terminology, code symbols, or proper nouns. BM25
    provides lexical keyword scoring based on term frequency (TF) and inverse document
    frequency (IDF).

    Features:
    - Tokenizes queries and documents with alphanumeric word extraction.
    - Computes BM25Okapi relevance scores for each indexed document.
    - Includes a term-frequency fallback for miniature test corpora (N <= 2) where
      standard BM25 IDF evaluates to 0.0, ensuring predictable ranking in all scenarios.
    - Returns the top `k` candidates paired with their scores.
    """

    def __init__(self, documents: List[Document], k: int = 20):
        self.k = k
        self.documents = documents
        if documents:
            from rank_bm25 import BM25Okapi
            self._tokenized_corpus = [
                re.findall(r"\w+", doc.page_content.lower()) for doc in documents
            ]
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """Scores documents against the query tokens and returns (Document, score) pairs."""
        if not self.documents or self._bm25 is None:
            return []
        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            return [(doc, 0.0) for doc in self.documents[: self.k]]
        scores = self._bm25.get_scores(tokens)
        if np.all(scores == 0.0):
            # Term-frequency fallback for edge-case small corpora
            scores = np.array([
                sum(doc_tokens.count(t) for t in tokens)
                for doc_tokens in self._tokenized_corpus
            ], dtype=float)
        top_indices = np.argsort(scores)[::-1][: self.k]
        return [(self.documents[i], float(scores[i])) for i in top_indices]

    def retrieve(self, query: str) -> List[Document]:
        return [doc for doc, _ in self.retrieve_with_scores(query)]

    def invoke(self, query: str) -> List[Document]:
        return self.retrieve(query)


class QueryFusionRetriever:
    """Multi-Query Expansion and Distance-Based Score Fusion Retriever.

    Designed for ambiguous, exploratory, or complex user inquiries. Often a single user prompt
    omits key vocabulary present in the indexed documents.

    Workflow:
    1. Uses an LLM with `query_gen_prompt` to generate 4 alternative query formulations.
    2. Combines the original query with the 4 generated variations (total 5 queries).
    3. Executes each query across all base retrievers (BM25 sparse + Dense vector store).
    4. Gathers all candidates and fuses their scores using `dist_based_score_fusion`
       with weights [0.4, 0.6] for BM25 and Vector store respectively.
    5. Returns the top `similarity_top_k` (default: 6) fused document chunks.
    """

    def __init__(
        self,
        retrievers: List[Any],
        retriever_weights: List[float] = [0.4, 0.6],
        llm: Any | None = None,
        query_gen_prompt: Any | None = None,
        similarity_top_k: int = 6,
        num_queries: int = 5,
        verbose: bool = True,
    ):
        self.retrievers = retrievers
        self.retriever_weights = retriever_weights
        self.llm = llm
        self.query_gen_prompt = query_gen_prompt
        self.similarity_top_k = similarity_top_k
        self.num_queries = num_queries
        self.verbose = verbose

    def _generate_queries(self, query: str) -> List[str]:
        """Prompts the LLM to rewrite the query into multiple complementary query angles."""
        if not self.llm or self.num_queries <= 1:
            return []
        try:
            formatted_prompt = self.query_gen_prompt.format(
                num_queries=self.num_queries, query=query
            )
            if hasattr(self.llm, "invoke"):
                res = self.llm.invoke(formatted_prompt)
                res_text = res.content if hasattr(res, "content") else str(res)
            elif hasattr(self.llm, "complete"):
                res_text = str(self.llm.complete(formatted_prompt))
            else:
                res_text = str(self.llm(formatted_prompt))

            lines = [
                re.sub(r"^\d+[\).\s\-*]+", "", line).strip()
                for line in res_text.strip().split("\n")
            ]
            queries = [line for line in lines if line and len(line) > 2]
            return queries[: self.num_queries]
        except Exception:
            return []

    def retrieve(self, query: str) -> List[ScoredDocument]:
        """Fuses multi-query expansion candidates across BM25 and Vector Store retrievers."""
        queries = [query] + self._generate_queries(query)
        all_candidates_with_weights = []

        for q in queries:
            for retriever, weight in zip(self.retrievers, self.retriever_weights):
                if hasattr(retriever, "retrieve_with_scores"):
                    candidates = retriever.retrieve_with_scores(q)
                elif hasattr(retriever, "similarity_search_with_score"):
                    candidates = retriever.similarity_search_with_score(q, k=20)
                elif hasattr(retriever, "invoke"):
                    docs = retriever.invoke(q)
                    candidates = [(d, float(len(docs) - i)) for i, d in enumerate(docs)]
                else:
                    candidates = []
                all_candidates_with_weights.append((candidates, weight))

        return dist_based_score_fusion(
            all_candidates_with_weights, top_k=self.similarity_top_k
        )

    def invoke(self, query: str) -> List[ScoredDocument]:
        return self.retrieve(query)


class TwoStageRetriever:
    """Two-Stage Hybrid Retriever with Cross-Encoder Neural Reranking.

    Designed for direct, specific queries where high precision is paramount.

    Architecture:
    - Stage 1 (Candidate Generation): Retrieves the top 20 candidates from both
      lexical (BM25) and semantic (Dense Vector) stores and combines them with
      distance-based score fusion (weights [0.4, 0.6]).
    - Stage 2 (Neural Reranking): Feeds all fused candidate chunks alongside the query
      into a cross-encoder model (`BAAI/bge-reranker-base`). The cross-encoder computes
      full token-to-token attention to re-rank candidates, producing the final top 6
      most relevant chunks with high precision.
    """

    def __init__(
        self,
        retrievers: List[Any],
        retriever_weights: List[float] = [0.4, 0.6],
        rerank_model_name: str = "BAAI/bge-reranker-base",
        similarity_top_k: int = 20,
        top_k_rerank: int = 6,
        verbose: bool = True,
    ):
        self.retrievers = retrievers
        self.retriever_weights = retriever_weights
        self.similarity_top_k = similarity_top_k
        self.top_k_rerank = top_k_rerank
        self.verbose = verbose
        self.reranker = CrossEncoderReranker(
            model_name=rerank_model_name, top_n=top_k_rerank
        )

    def retrieve(self, query: str) -> List[ScoredDocument]:
        """Executes Stage 1 hybrid candidate retrieval followed by Stage 2 neural reranking."""
        all_candidates_with_weights = []
        for retriever, weight in zip(self.retrievers, self.retriever_weights):
            if hasattr(retriever, "retrieve_with_scores"):
                candidates = retriever.retrieve_with_scores(query)
            elif hasattr(retriever, "similarity_search_with_score"):
                candidates = retriever.similarity_search_with_score(
                    query, k=self.similarity_top_k
                )
            elif hasattr(retriever, "invoke"):
                docs = retriever.invoke(query)
                candidates = [(d, float(len(docs) - i)) for i, d in enumerate(docs)]
            else:
                candidates = []
            all_candidates_with_weights.append((candidates, weight))

        # Stage 1: Distance-based score fusion across BM25 and Dense candidates
        fused_candidates = dist_based_score_fusion(
            all_candidates_with_weights, top_k=self.similarity_top_k
        )

        # Stage 2: Cross-encoder neural reranking to top_k_rerank (default: 6)
        return self.reranker.rerank(query, fused_candidates)

    def invoke(self, query: str) -> List[ScoredDocument]:
        return self.retrieve(query)


class RouterRetriever:
    """Intelligent Retrieval Router using LLM Query Classification.

    Evaluates user query clarity and dynamically selects the optimal retrieval pipeline:
    - Choice 1 (Query Fusion): Triggered when the query is ambiguous, broad, or lacks specifics.
      Expands the query into multiple sub-queries to broaden recall.
    - Choice 2 (Two-Stage Rerank): Triggered when the query is clear and specific.
      Leverages cross-encoder joint attention for pinpoint precision.
    """

    def __init__(
        self,
        fusion_retriever: QueryFusionRetriever,
        two_stage_retriever: TwoStageRetriever,
        llm: Any | None = None,
        language: str = "eng",
    ):
        self.fusion_retriever = fusion_retriever
        self.two_stage_retriever = two_stage_retriever
        self.llm = llm
        self.language = language

    def retrieve(self, query: str) -> List[ScoredDocument]:
        """Uses LLM classification to route to either Fusion Retriever or Two-Stage Retriever."""
        if not self.llm:
            return self.two_stage_retriever.retrieve(query)

        prompt_tmpl = get_single_select_prompt(self.language)
        choices_text = (
            "(1) Fusion Retriever with BM25 and Vector Retriever and LLM Query Generation. Use this tool when the user's query is ambiguous or unclear.\n"
            "(2) Two Stage Retriever with BM25 and Vector Retriever and LLM Rerank. Use this tool when the user's query is clear and unambiguous."
        )
        prompt_str = prompt_tmpl.format(
            num_choices=2,
            context_list=choices_text,
            query_str=query,
        )

        try:
            if hasattr(self.llm, "invoke"):
                res = self.llm.invoke(prompt_str)
                res_text = res.content if hasattr(res, "content") else str(res)
            elif hasattr(self.llm, "complete"):
                res_text = str(self.llm.complete(prompt_str))
            else:
                res_text = str(self.llm(prompt_str))

            # Parse selection choice from model response
            choice = 2  # default to clear / two-stage
            if "1" in res_text and "2" not in res_text:
                choice = 1
            elif "2" in res_text:
                choice = 2

            if choice == 1:
                return self.fusion_retriever.retrieve(query)
            else:
                return self.two_stage_retriever.retrieve(query)
        except Exception:
            return self.two_stage_retriever.retrieve(query)

    def invoke(self, query: str) -> List[ScoredDocument]:
        return self.retrieve(query)


class LocalRetriever:
    """Retriever Assembly Factory and Conditional Routing Manager.

    Constructs and configures the appropriate retrieval strategy based on document volume:
    - Small Collections (<= top_k_rerank nodes, default 6): Employs direct dense vector similarity
      to avoid unnecessary reranker computational overhead.
    - Large Collections (> top_k_rerank nodes): Assembles the full hybrid routing pipeline
      (Dense VectorStore + BM25 + RouterRetriever choosing between Query Fusion and TwoStage Rerank).
    """

    def __init__(
        self, setting: RAGSettings | None = None, host: str = "host.docker.internal"
    ):
        self._setting = setting or RAGSettings()
        self._host = host
        self._vector_store_factory = LocalVectorStore(host=host, setting=self._setting)

    def _get_normal_retriever(
        self,
        vector_index: Any,
        llm: Any | None = None,
        language: str = "eng",
    ):
        """Standard dense vector similarity retriever for collections with <= top_k_rerank nodes."""
        class _NormalRetriever:
            def __init__(self, index, top_k):
                self.index = index
                self.top_k = top_k

            def retrieve(self, query: str) -> List[ScoredDocument]:
                if hasattr(self.index, "similarity_search_with_score"):
                    results = self.index.similarity_search_with_score(query, k=self.top_k)
                    return [ScoredDocument(doc, score) for doc, score in results]
                elif hasattr(self.index, "similarity_search"):
                    results = self.index.similarity_search(query, k=self.top_k)
                    return [ScoredDocument(doc, 1.0) for doc in results]
                return []

            def invoke(self, query: str) -> List[ScoredDocument]:
                return self.retrieve(query)

        return _NormalRetriever(
            vector_index, self._setting.retriever.similarity_top_k
        )

    def _get_hybrid_retriever(
        self,
        vector_index: Any,
        bm25_retriever: LangChainBM25Retriever,
        llm: Any | None = None,
        language: str = "eng",
        gen_query: bool = True,
    ):
        """Constructs a hybrid retriever combining Vector + BM25."""
        if gen_query:
            return QueryFusionRetriever(
                retrievers=[bm25_retriever, vector_index],
                retriever_weights=self._setting.retriever.retriever_weights,
                llm=llm,
                query_gen_prompt=get_query_gen_prompt(language),
                similarity_top_k=self._setting.retriever.top_k_rerank,
                num_queries=self._setting.retriever.num_queries,
                verbose=True,
            )
        else:
            return TwoStageRetriever(
                retrievers=[bm25_retriever, vector_index],
                retriever_weights=self._setting.retriever.retriever_weights,
                rerank_model_name=self._setting.retriever.rerank_llm,
                similarity_top_k=self._setting.retriever.similarity_top_k,
                top_k_rerank=self._setting.retriever.top_k_rerank,
                verbose=True,
            )

    def _get_router_retriever(
        self,
        vector_index: Any,
        bm25_retriever: LangChainBM25Retriever,
        llm: Any | None = None,
        language: str = "eng",
    ):
        """Uses LLM selector to route between Fusion Query Generation and Two-Stage Rerank."""
        fusion_retriever = self._get_hybrid_retriever(
            vector_index, bm25_retriever, llm, language, gen_query=True
        )
        two_stage_retriever = self._get_hybrid_retriever(
            vector_index, bm25_retriever, llm, language, gen_query=False
        )
        return RouterRetriever(
            fusion_retriever=fusion_retriever,
            two_stage_retriever=two_stage_retriever,
            llm=llm,
            language=language,
        )

    def get_retrievers(
        self,
        nodes: List[Document],
        llm: Any | None = None,
        language: str = "eng",
        embed_model: Any | None = None,
    ):
        """Constructs in-memory VectorStore and selects router vs standard retriever based on node count."""
        vector_index = self._vector_store_factory.get_index(nodes, embed_model=embed_model)
        bm25_retriever = LangChainBM25Retriever(
            nodes, k=self._setting.retriever.similarity_top_k
        )

        # For larger document sets (> top_k_rerank), use routed hybrid retrieval; otherwise simple dense
        if len(nodes) > self._setting.retriever.top_k_rerank:
            retriever = self._get_router_retriever(
                vector_index, bm25_retriever, llm, language
            )
        else:
            retriever = self._get_normal_retriever(vector_index, llm, language)

        return retriever
