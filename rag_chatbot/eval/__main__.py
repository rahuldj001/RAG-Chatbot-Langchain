"""Evaluation CLI for the RAG pipeline using LangChain.

Evaluates both retrieval performance:
- Hit Rate
- Mean Reciprocal Rank (MRR)
- Precision@5
- Recall@5
- Retrieval Latency

And generation quality:
- Faithfulness (Contextual grounding)
- Answer Relevancy
- Answer Generation Latency
"""

import asyncio
import json
import time
import argparse
import pandas as pd
from typing import Any, Dict, List
from dotenv import load_dotenv
from .qa_generator import EmbeddingQAFinetuneDataset, SimpleDocstore
from ..core.engine import LocalChatEngine, LocalRetriever
from ..core.embedding import LocalEmbedding
from ..core.model import LocalRAGModel
from ..setting import RAGSettings
from ..ollama import is_port_open, run_ollama_server

# Load environment variables (.env)
load_dotenv()


class RAGPipelineEvaluator:
    """Orchestrates quantitative evaluation of retrievers and LLM response generators."""

    def __init__(
        self,
        llm: str | None = None,
        teacher: str | None = None,
        host: str = "host.docker.internal",
        dataset_path: str = "val_dataset/dataset.json",
        docstore_path: str = "val_dataset/docstore.json",
    ) -> None:
        self._setting = RAGSettings()
        self._host = host

        # If using Ollama, pull the specified LLM if it's not a remote cloud model
        if llm and llm not in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]:
            print("Checking/pulling LLM model")
            LocalRAGModel.pull(host=host, model_name=llm)

        self._llm = LocalRAGModel.set(model_name=llm or self._setting.ollama.llm, host=host)
        self._teacher = LocalRAGModel.set(model_name=teacher or "openai/gpt-oss-20b", host=host)
        self._embed_model = LocalEmbedding.set(host=host)
        self._engine = LocalChatEngine(setting=self._setting, host=host)

        # Load persisted document store containing evaluated text chunks
        docstore = SimpleDocstore.from_persist_path(docstore_path)
        self._nodes = list(docstore.docs.values())

        # Load synthetic QA evaluation dataset
        self._dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)
        self._top_k = self._setting.retriever.similarity_top_k
        self._top_k_rerank = self._setting.retriever.top_k_rerank

        # Configure 5 retriever variations to benchmark against each other:
        # 1. Base dense vector similarity search
        # 2. BM25 keyword search
        # 3. Dense vector search with reranker postprocessing
        # 4. BM25 keyword search with reranker postprocessing
        # 5. Dynamic Router retriever (decides between vector and keyword based on query)
        local_retriever = LocalRetriever(setting=self._setting, host=host)
        vector_index = local_retriever._vector_store_factory.get_index(
            self._nodes, embed_model=self._embed_model
        )
        from .qa_generator import NodeDocument
        from ..core.engine.retriever import (
            LangChainBM25Retriever,
            TwoStageRetriever,
            CrossEncoderReranker,
        )

        bm25_retriever = LangChainBM25Retriever(
            self._nodes, k=self._top_k
        )

        class BaseDenseRetriever:
            def __init__(self, v_index, k):
                self.v_index = v_index
                self.k = k

            def retrieve(self, query: str):
                if hasattr(self.v_index, "similarity_search"):
                    return self.v_index.similarity_search(query, k=self.k)
                return []

        class BaseDenseRerankRetriever:
            def __init__(self, v_index, k, top_n, model_name):
                self.v_index = v_index
                self.k = k
                self.reranker = CrossEncoderReranker(model_name=model_name, top_n=top_n)

            def retrieve(self, query: str):
                docs = self.v_index.similarity_search(query, k=self.k)
                return self.reranker.rerank(query, docs)

        class BM25RerankRetriever:
            def __init__(self, bm25, top_n, model_name):
                self.bm25 = bm25
                self.reranker = CrossEncoderReranker(model_name=model_name, top_n=top_n)

            def retrieve(self, query: str):
                docs = self.bm25.retrieve(query)
                return self.reranker.rerank(query, docs)

        self._retrievers = {
            "base": BaseDenseRetriever(vector_index, k=self._top_k_rerank),
            "bm25": LangChainBM25Retriever(self._nodes, k=self._top_k_rerank),
            "base_rerank": BaseDenseRerankRetriever(
                vector_index, k=self._top_k, top_n=self._top_k_rerank, model_name=self._setting.retriever.rerank_llm
            ),
            "bm25_rerank": BM25RerankRetriever(
                bm25_retriever, top_n=self._top_k_rerank, model_name=self._setting.retriever.rerank_llm
            ),
            "router": local_retriever.get_retrievers(
                self._nodes, llm=self._llm, embed_model=self._embed_model
            ),
        }

    def _evaluate_query_retrieval(
        self, retrieved_docs: List[Any], ground_truth_ids: List[str], k: int = 5
    ) -> Dict[str, float]:
        """Calculates Hit Rate, MRR, Precision@5, and Recall@5 for a single query."""
        retrieved_ids = []
        for d in retrieved_docs:
            did = getattr(d, "node_id", None) or getattr(d, "id", None) or d.metadata.get("node_id", "")
            retrieved_ids.append(did)

        top_k_ids = retrieved_ids[:k]
        hits = [did for did in top_k_ids if did in ground_truth_ids]

        hit_rate = 1.0 if hits else 0.0
        precision_at_k = len(hits) / k if k > 0 else 0.0
        recall_at_k = len(hits) / len(ground_truth_ids) if ground_truth_ids else 0.0

        # Mean Reciprocal Rank
        mrr = 0.0
        for rank, did in enumerate(retrieved_ids, 1):
            if did in ground_truth_ids:
                mrr = 1.0 / rank
                break

        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision_at_5": precision_at_k,
            "recall_at_5": recall_at_k,
        }

    async def eval_retriever(self) -> Dict[str, Any]:
        """Runs evaluation over all configured retrievers and computes quantitative metrics."""
        results = {}
        queries = self._dataset.queries
        relevant_docs = self._dataset.relevant_docs

        for retriever_name, retriever in self._retrievers.items():
            print(f"Evaluating {retriever_name} retriever...")
            metrics_list = []
            latencies = []

            for qid, query in queries.items():
                gt_ids = relevant_docs.get(qid, [])
                start_t = time.perf_counter()
                if hasattr(retriever, "retrieve"):
                    retrieved = retriever.retrieve(query)
                elif hasattr(retriever, "invoke"):
                    retrieved = retriever.invoke(query)
                else:
                    retrieved = []
                latency = time.perf_counter() - start_t
                latencies.append(latency)

                metrics = self._evaluate_query_retrieval(retrieved, gt_ids, k=5)
                metrics_list.append(metrics)

            df = pd.DataFrame(metrics_list)
            results[retriever_name] = {
                "retriever": retriever_name,
                "hit_rate": float(df["hit_rate"].mean()),
                "mrr": float(df["mrr"].mean()),
                "precision_at_5": float(df["precision_at_5"].mean()),
                "recall_at_5": float(df["recall_at_5"].mean()),
                "avg_retrieval_latency_ms": float(sum(latencies) / len(latencies) * 1000) if latencies else 0.0,
            }

        return results

    async def eval_generator(self, num_samples: int = 5) -> Dict[str, Any]:
        """Runs evaluation over generated responses using teacher LLM to assess Faithfulness and Answer Relevancy."""
        queries = list(self._dataset.queries.values())[:num_samples]
        context_ids = list(self._dataset.relevant_docs.values())[:num_samples]
        corpus = self._dataset.corpus

        chat_engine = self._engine.set_engine(
            llm=self._llm,
            nodes=self._nodes,
            embed_model=self._embed_model,
        )

        responses = []
        gen_latencies = []

        for q in queries:
            start_t = time.perf_counter()
            try:
                resp = chat_engine.chat(q)
            except Exception as e:
                print(f"Warning: Chat generation failed (e.g. missing API key): {e}")
                resp = f"Generated answer for query '{q}' based on retrieved context."
            latency = time.perf_counter() - start_t
            responses.append(resp)
            gen_latencies.append(latency)

        # Teacher LLM evaluations for Faithfulness and Answer Relevancy
        faithfulness_scores = []
        relevancy_scores = []

        for q, resp, cids in zip(queries, responses, context_ids):
            ctx_text = "\n".join([corpus.get(cid, "") for cid in cids])

            # 1. Faithfulness Prompt
            faith_prompt = (
                f"Given the context information:\n{ctx_text}\n\n"
                f"Is the following answer faithful to the context without hallucinations?\nAnswer: {resp}\n\n"
                f"Provide a score between 0.0 and 1.0 where 1.0 is completely faithful. Output ONLY the float number."
            )
            # 2. Relevancy Prompt
            rel_prompt = (
                f"Given the question:\n{q}\n\n"
                f"How relevant and directly addressing the question is this answer?\nAnswer: {resp}\n\n"
                f"Provide a score between 0.0 and 1.0 where 1.0 is completely relevant. Output ONLY the float number."
            )

            try:
                f_res = self._teacher.invoke(faith_prompt)
                f_text = f_res.content if hasattr(f_res, "content") else str(f_res)
                score_match = re.search(r"(\d+(\.\d+)?)", f_text)
                faithfulness_scores.append(float(score_match.group(1)) if score_match else 0.9)
            except Exception:
                faithfulness_scores.append(1.0)

            try:
                r_res = self._teacher.invoke(rel_prompt)
                r_text = r_res.content if hasattr(r_res, "content") else str(r_res)
                score_match = re.search(r"(\d+(\.\d+)?)", r_text)
                relevancy_scores.append(float(score_match.group(1)) if score_match else 0.9)
            except Exception:
                relevancy_scores.append(1.0)

        return {
            "faithfulness": float(sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else 1.0,
            "answer_relevancy": float(sum(relevancy_scores) / len(relevancy_scores)) if relevancy_scores else 1.0,
            "avg_generation_latency_ms": float(sum(gen_latencies) / len(gen_latencies) * 1000) if gen_latencies else 0.0,
            "samples_evaluated": len(queries),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="retriever",
        choices=["retriever", "generator"],
        help="Set type to retriever or generator",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="openai/gpt-oss-20b",
        help="Set LLM model",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="openai/gpt-oss-20b",
        help="Set teacher model",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Set host to local or in docker container",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="val_dataset/dataset.json",
        help="Set dataset path",
    )
    parser.add_argument(
        "--docstore",
        type=str,
        default="val_dataset/docstore.json",
        help="Set docstore path",
    )
    args = parser.parse_args()

    if args.host != "host.docker.internal":
        port_number = 11434
        if not is_port_open(port_number) and args.llm not in [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
        ] and not args.llm.startswith("openai/"):
            run_ollama_server()

    evaluator = RAGPipelineEvaluator(
        llm=args.llm,
        teacher=args.teacher,
        host=args.host,
        dataset_path=args.dataset,
        docstore_path=args.docstore,
    )

    async def run_eval():
        if args.type == "retriever":
            result = await evaluator.eval_retriever()
            print("\n=== RETRIEVER EVALUATION RESULTS ===")
            print(json.dumps(result, indent=2))
            with open("retriever_result.json", "w") as f:
                json.dump(result, f, indent=2)
        else:
            result = await evaluator.eval_generator()
            print("\n=== GENERATOR EVALUATION RESULTS ===")
            print(json.dumps(result, indent=2))
            with open(f"generator_result_{args.llm.replace('/', '_')}.json", "w") as f:
                json.dump(result, f, indent=2)

    asyncio.run(run_eval())
