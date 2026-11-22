"""Data Ingestion pipeline for parsing, chunking, and embedding PDF documents.

Extracts text from uploaded PDF files using PyMuPDF (fitz), sanitizes the text,
splits long documents into coherent semantic chunks (nodes) with SentenceSplitter,
and computes dense vector embeddings for indexing into the LangChain RAG retriever.
"""

import os
import re
import uuid
import pymupdf as fitz  # PyMuPDF: high-performance PDF text extraction
from dotenv import load_dotenv
from typing import Any, Callable, List, Optional
from tqdm import tqdm
from langchain_core.documents import Document
from ...setting import RAGSettings

load_dotenv()


class NodeDocument(Document):
    """Specialized document chunk representation extending LangChain's Document.

    In a Retrieval-Augmented Generation (RAG) system, raw documents are split into smaller
    atomic units ('nodes' or 'chunks'). This class encapsulates each chunk's text and metadata
    while exposing convenient property accessors:
    - `.text` / `.page_content`: Raw text content of the chunk.
    - `.node_id` / `.id`: Globally unique identifier for retrieval citation and tracking.
    - `.embedding`: Pre-computed dense embedding vector stored directly on metadata.
    - `.get_content()`: Helper method returning the chunk's text for downstream chains.
    """
    model_config = {"extra": "allow"}

    def __init__(self, page_content: str, metadata: dict | None = None, id: str | None = None, **kwargs):
        super().__init__(page_content=page_content, metadata=metadata or {}, id=id, **kwargs)
        # Ensure node_id is bidirectional between top-level id and metadata dictionary
        if "node_id" not in self.metadata and self.id:
            self.metadata["node_id"] = self.id
        elif "node_id" in self.metadata and not self.id:
            self.id = self.metadata["node_id"]

    @property
    def text(self) -> str:
        """Convenience property returning the text content of this chunk."""
        return self.page_content

    @text.setter
    def text(self, value: str):
        self.page_content = value

    @property
    def node_id(self) -> str:
        """Unique ID string for this document chunk."""
        return self.id or self.metadata.get("node_id", "")

    @node_id.setter
    def node_id(self, value: str):
        self.id = value
        self.metadata["node_id"] = value

    @property
    def embedding(self) -> list[float] | None:
        """The dense vector embedding of this chunk, if already computed."""
        return self.metadata.get("embedding", None)

    @embedding.setter
    def embedding(self, value: list[float]):
        self.metadata["embedding"] = value

    def get_content(self, metadata_mode: Any = None) -> str:
        """Returns the chunk text content."""
        return self.page_content


class SentenceSplitter:
    """Semantic sentence-aware text splitter for RAG chunking.

    Splits large text documents into coherent, contextually bounded chunks:
    1. First splits text into high-level blocks using paragraph separators (e.g. `\n \n`).
    2. Then segments paragraphs into fine-grained sentences using regular expressions
       targeting terminal punctuation (e.g. `.`, `;`, `!`, `?`, and Asian equivalents `。？！`).
    3. Assembles sentences sequentially into chunks up to `chunk_size` (default: 512 tokens).
    4. Carries over `chunk_overlap` (default: 32 tokens) trailing sentences into the next
       chunk to prevent loss of semantic context across chunk boundaries.
    5. Safely fragments single sentences that exceed `chunk_size` at word boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 32,
        paragraph_separator: str = "\n \n",
        secondary_chunking_regex: str = "[^,.;。？！]+[,.;。？！]?",
        tokenizer: Optional[Callable[[str], List[Any]]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.paragraph_separator = paragraph_separator
        self.secondary_chunking_regex = secondary_chunking_regex

        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                self._tokenizer = enc.encode
            except Exception:
                self._tokenizer = lambda s: s.split()

    def _token_len(self, text: str) -> int:
        return len(self._tokenizer(text))

    def _split_into_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        sentences = re.findall(self.secondary_chunking_regex, text)
        if not sentences:
            sentences = [text]
        return [s.strip() for s in sentences if s.strip()]

    def split_text(self, text: str) -> List[str]:
        """Splits raw text into a list of chunks matching chunk_size and overlap."""
        if not text or not text.strip():
            return []

        paragraphs = text.split(self.paragraph_separator)
        all_sentences: List[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_sentences = self._split_into_sentences(para)
            all_sentences.extend(para_sentences)

        if not all_sentences:
            return []

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_len = 0

        for sentence in all_sentences:
            sentence_len = self._token_len(sentence)
            if sentence_len > self.chunk_size:
                # Break unusually long sentences into words
                words = sentence.split()
                sub_chunk: List[str] = []
                sub_len = 0
                for w in words:
                    w_len = self._token_len(w)
                    if sub_len + w_len > self.chunk_size and sub_chunk:
                        chunk_str = " ".join(sub_chunk)
                        chunks.append(chunk_str)
                        # Overlap
                        overlap_words = sub_chunk[-max(1, self.chunk_overlap // 4):]
                        sub_chunk = list(overlap_words)
                        sub_len = self._token_len(" ".join(sub_chunk))
                    sub_chunk.append(w)
                    sub_len += w_len
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                continue

            if current_len + sentence_len > self.chunk_size and current_sentences:
                chunk_str = " ".join(current_sentences)
                chunks.append(chunk_str)

                # Collect overlapping trailing sentences
                overlap_sentences: List[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    s_len = self._token_len(s)
                    if overlap_len + s_len <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += s_len
                    else:
                        break
                current_sentences = overlap_sentences
                current_len = overlap_len

            current_sentences.append(sentence)
            current_len += sentence_len

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def __call__(self, documents: List[Document], show_progress: bool = False) -> List[NodeDocument]:
        """Takes a list of Documents and returns a list of chunked NodeDocuments."""
        nodes: List[NodeDocument] = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for idx, chunk in enumerate(chunks):
                node_id = str(uuid.uuid4())
                meta = dict(doc.metadata)
                meta["node_id"] = node_id
                meta["chunk_index"] = idx
                node = NodeDocument(page_content=chunk, metadata=meta, id=node_id)
                nodes.append(node)
        return nodes

    @classmethod
    def from_defaults(
        cls,
        chunk_size: int = 512,
        chunk_overlap: int = 32,
        paragraph_separator: str = "\n \n",
        secondary_chunking_regex: str = "[^,.;。？！]+[,.;。？！]?",
    ):
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator=paragraph_separator,
            secondary_chunking_regex=secondary_chunking_regex,
        )


class LocalDataIngestion:
    def __init__(self, setting: RAGSettings | None = None) -> None:
        self._setting = setting or RAGSettings()
        # In-memory cache mapping {file_name: list_of_embedded_nodes} to avoid re-parsing
        self._node_store: dict[str, List[NodeDocument]] = {}
        # Tracks files currently active in the chat session
        self._ingested_file: list[str] = []

    def _filter_text(self, text: str) -> str:
        """Sanitizes raw extracted PDF text.

        Removes non-printable or corrupt control characters while retaining alphanumeric
        content, Vietnamese/accented characters, common punctuation, and normalizing whitespace.
        """
        pattern = r'[a-zA-Z0-9 \u00C0-\u01B0\u1EA0-\u1EF9`~!@#$%^&*()_\-+=\[\]{}|\\;:\'",.<>/?]+'
        matches = re.findall(pattern, text)
        filtered_text = " ".join(matches)
        # Collapse multiple spaces, tabs, and newlines into single spaces
        normalized_text = re.sub(r"\s+", " ", filtered_text.strip())
        return normalized_text

    def store_nodes(
        self,
        input_files: list[str],
        embed_nodes: bool = True,
        embed_model: Any | None = None,
    ) -> List[NodeDocument]:
        """Parses a list of PDF files, chunks them into nodes, and computes embeddings."""
        return_nodes: List[NodeDocument] = []
        self._ingested_file = []
        if len(input_files) == 0:
            return return_nodes

        splitter = SentenceSplitter.from_defaults(
            chunk_size=self._setting.ingestion.chunk_size,
            chunk_overlap=self._setting.ingestion.chunk_overlap,
            paragraph_separator=self._setting.ingestion.paragraph_sep,
            secondary_chunking_regex=self._setting.ingestion.chunking_regex,
        )

        for input_file in tqdm(input_files, desc="Ingesting data"):
            # Normalize path basename across Windows/Linux delimiters
            file_name = os.path.basename(input_file.strip())
            self._ingested_file.append(file_name)

            # Check if this document was already parsed and embedded
            if file_name in self._node_store:
                return_nodes.extend(self._node_store[file_name])
            else:
                # Open PDF with PyMuPDF and extract text page by page
                document = fitz.open(input_file)
                all_text = ""
                for doc_idx, page in enumerate(document):
                    page_text = page.get_text("text")
                    page_text = self._filter_text(page_text)
                    all_text += " " + page_text

                # Wrap parsed text into a Document object with metadata
                doc = Document(
                    page_content=all_text.strip(),
                    metadata={
                        "file_name": file_name,
                    },
                )

                # Chunk document into smaller NodeDocument objects
                nodes = splitter([doc], show_progress=True)

                # Compute dense vector embeddings for each chunk if requested
                if embed_nodes and embed_model is not None:
                    texts = [n.page_content for n in nodes]
                    if hasattr(embed_model, "embed_documents"):
                        embeddings = embed_model.embed_documents(texts)
                        for n, emb in zip(nodes, embeddings):
                            n.embedding = emb
                    elif callable(embed_model):
                        # Support custom callable embedding functions
                        nodes = embed_model(nodes)

                # Cache embedded nodes in memory for fast retrieval
                self._node_store[file_name] = nodes
                return_nodes.extend(nodes)

        return return_nodes

    def reset(self):
        """Clears all cached nodes and active file lists."""
        self._node_store = {}
        self._ingested_file = []

    def check_nodes_exist(self) -> bool:
        """Returns True if any documents have been ingested and are ready for querying."""
        return len(self._node_store.values()) > 0

    def get_all_nodes(self) -> List[NodeDocument]:
        """Returns all ingested nodes across all documents currently stored."""
        return_nodes = []
        for nodes in self._node_store.values():
            return_nodes.extend(nodes)
        return return_nodes

    def get_ingested_nodes(self) -> List[NodeDocument]:
        """Returns nodes belonging to currently active documents."""
        return_nodes = []
        for file in self._ingested_file:
            if file in self._node_store:
                return_nodes.extend(self._node_store[file])
        return return_nodes
