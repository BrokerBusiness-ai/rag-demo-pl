"""rag-demo-pl, RAG na polskim korpusie, od chunkingu po ewaluację."""

from .chunking import (
    Chunk,
    Chunker,
    FixedSizeChunker,
    MarkdownChunker,
    SentenceChunker,
)
from .embeddings import (
    CachedEmbedder,
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)
from .evaluation import (
    EvalCase,
    EvalResult,
    compare_modes,
    evaluate,
    format_comparison,
)
from .pipeline import DEFAULT_PROMPT, RAGPipeline, RAGResponse
from .store import BM25, SearchResult, VectorStore

__version__ = "0.1.0"

__all__ = [
    # Chunking
    "Chunk",
    "Chunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "MarkdownChunker",
    # Embeddings
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "CachedEmbedder",
    # Store
    "VectorStore",
    "SearchResult",
    "BM25",
    # Pipeline
    "RAGPipeline",
    "RAGResponse",
    "DEFAULT_PROMPT",
    # Evaluation
    "EvalCase",
    "EvalResult",
    "evaluate",
    "compare_modes",
    "format_comparison",
]
