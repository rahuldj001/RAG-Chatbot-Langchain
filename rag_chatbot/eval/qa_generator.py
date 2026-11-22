"""Synthetic QA Dataset Generator for RAG Evaluation and Fine-Tuning using LangChain.

Uses an LLM (acting as a professor) to generate evaluation questions from ingested document chunks.
"""

import json
import random
import re
import os
import uuid
from typing import Any, Dict, List
from tqdm import tqdm
from langchain_core.documents import Document
from ..core.model import LocalRAGModel
from ..core.embedding import LocalEmbedding
from ..core.ingestion import LocalDataIngestion, NodeDocument
from ..setting import RAGSettings

# Prompt template instructing the LLM to act as a teacher and formulate quiz questions
DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the context information provided. \
Only provide the questions, not the answers."
"""


class EmbeddingQAFinetuneDataset:
    """Dataset container holding synthetic queries, document corpus, and relevance mappings."""

    def __init__(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        relevant_docs: Dict[str, List[str]],
    ):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs

    def save_json(self, path: str):
        data = {
            "queries": self.queries,
            "corpus": self.corpus,
            "relevant_docs": self.relevant_docs,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            queries=data.get("queries", {}),
            corpus=data.get("corpus", {}),
            relevant_docs=data.get("relevant_docs", {}),
        )


class SimpleDocstore:
    """Simple document store for saving and loading evaluated text chunks."""

    def __init__(self, docs: Dict[str, Document] | None = None):
        self.docs: Dict[str, Document] = docs or {}

    def add_documents(self, documents: List[Document]):
        for doc in documents:
            doc_id = getattr(doc, "node_id", None) or getattr(doc, "id", None) or doc.metadata.get("node_id") or str(uuid.uuid4())
            self.docs[doc_id] = doc

    def persist(self, persist_path: str):
        serialized = {}
        for did, doc in self.docs.items():
            serialized[did] = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "id": getattr(doc, "id", did),
            }
        with open(persist_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_persist_path(cls, persist_path: str):
        with open(persist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = {}
        for did, d in data.items():
            docs[did] = NodeDocument(
                page_content=d.get("page_content", ""),
                metadata=d.get("metadata", {}),
                id=d.get("id", did),
            )
        return cls(docs)


def generate_question_context_pairs(
    nodes: List[Document],
    llm: Any,
    qa_generate_prompt_tmpl: str = DEFAULT_QA_GENERATE_PROMPT_TMPL,
    num_questions_per_chunk: int = 2,
) -> EmbeddingQAFinetuneDataset:
    """Generate question-context pairs given a list of parsed text nodes."""
    node_dict = {
        (getattr(node, "node_id", None) or getattr(node, "id", None) or node.metadata.get("node_id", str(idx))): node.page_content
        for idx, node in enumerate(nodes)
    }

    queries = {}
    relevant_docs = {}

    for node_id, text in tqdm(node_dict.items(), desc="Generating questions"):
        query_prompt = qa_generate_prompt_tmpl.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )

        try:
            if hasattr(llm, "invoke"):
                res = llm.invoke(query_prompt)
                res_text = res.content if hasattr(res, "content") else str(res)
            elif hasattr(llm, "complete"):
                res_text = str(llm.complete(query_prompt))
            else:
                res_text = str(llm(query_prompt))
        except Exception as e:
            print(f"Error generating question for node {node_id}: {e}")
            continue

        result = str(res_text).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s\-*]+", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 5 and "?" in question or len(question) > 10]

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]

    return EmbeddingQAFinetuneDataset(
        queries=queries, corpus=node_dict, relevant_docs=relevant_docs
    )


class QAGenerator:
    """Coordinates document ingestion, chunking, and synthetic evaluation question generation."""

    def __init__(
        self,
        embed_model: str | None = None,
        llm: str | None = None,
        host: str = "host.docker.internal",
    ) -> None:
        setting = RAGSettings()
        setting.ingestion.embed_llm = embed_model or setting.ingestion.embed_llm
        self._embed_model = LocalEmbedding.set(setting)
        self._llm = LocalRAGModel.set(model_name=llm or setting.ollama.llm, host=host)
        self._ingestion = LocalDataIngestion()

    def generate(
        self,
        input_files: list[str],
        output_dir: str = "val_dataset",
        max_nodes: int = 100,
        num_questions_per_chunk: int = 2,
    ) -> None:
        """Ingests given input files, samples nodes, generates questions, and saves dataset to disk."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(os.path.join(output_dir, "docstore.json")):
            print("Docstore already exists! Skip ingestion.")

        nodes = self._ingestion.store_nodes(input_files, embed_nodes=True, embed_model=self._embed_model)
        random.shuffle(nodes)

        dataset = generate_question_context_pairs(
            nodes=nodes[:max_nodes],
            llm=self._llm,
            num_questions_per_chunk=num_questions_per_chunk,
        )

        dataset.save_json(os.path.join(output_dir, "dataset.json"))

        docstore = SimpleDocstore()
        docstore.add_documents(nodes)
        docstore.persist(persist_path=os.path.join(output_dir, "docstore.json"))
