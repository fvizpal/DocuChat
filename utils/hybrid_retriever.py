
import math
from rank_bm25 import BM25Okapi
from utils.indexer import embed_query, _get_collection
from utils.loader import Document


# ---------------------------------------------------------------------------
# BM25 index is built in-memory from whatever is currently in ChromaDB.
# It rebuilds automatically whenever called, so it always stays in sync
# with the vector index — no separate BM25 persistence needed.
# ---------------------------------------------------------------------------

def _build_bm25_index() -> tuple[BM25Okapi, list[str], list[dict]]:
    """
    Pull all stored chunks from ChromaDB and build a BM25 index
    over them in memory.

    Returns:
        bm25       — the BM25 index ready for searching
        all_texts  — original text of every chunk (same order as BM25)
        all_metas  — metadata of every chunk (same order as BM25)
    """
    collection = _get_collection()

    if collection.count() == 0:
        raise ValueError(
            "No documents indexed yet. "
            "Please load and index a document first."
        )

    # Fetch ALL chunks from ChromaDB (no query — just get everything)
    all_data = collection.get(include=["documents", "metadatas"])

    all_texts = all_data["documents"]
    all_metas = all_data["metadatas"]

    # BM25 works on tokenized text — simple whitespace tokenization
    # is good enough here; we could use a proper tokenizer later
    tokenized = [text.lower().split() for text in all_texts]
    bm25 = BM25Okapi(tokenized)

    return bm25, all_texts, all_metas


def _reciprocal_rank_fusion(
    dense_chunks: list[Document],
    sparse_chunks: list[Document],
    k: int = 60          # RRF smoothing constant — 60 is the standard default
) -> list[Document]:
    """
    Merge dense (semantic) and sparse (BM25) result lists using
    Reciprocal Rank Fusion.

    RRF score for a chunk = 1/(k + rank_in_dense) + 1/(k + rank_in_sparse)

    Why RRF instead of combining raw scores directly?
    Because cosine similarity (0-1) and BM25 scores (unbounded floats)
    are on completely different scales — you can't add them meaningfully.
    RRF only uses rank positions, which are always comparable.
    (Lesson 6: this is exactly the scale mismatch problem RRF solves)
    """

    # Map chunk text → RRF score (use text as unique key)
    rrf_scores: dict[str, float] = {}

    # Map chunk text → Document object (to reconstruct results)
    chunk_map: dict[str, Document] = {}

    # Score from dense retrieval ranks
    for rank, chunk in enumerate(dense_chunks, start=1):
        key = chunk.text
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
        chunk_map[key] = chunk

    # Score from sparse (BM25) retrieval ranks
    for rank, chunk in enumerate(sparse_chunks, start=1):
        key = chunk.text
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
        chunk_map[key] = chunk

    # Sort by combined RRF score descending
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    # Reconstruct Document list with RRF score in metadata
    fused = []
    for key in sorted_keys:
        chunk = chunk_map[key]
        fused.append(Document(
            text=chunk.text,
            metadata={
                **chunk.metadata,
                "rrf_score": round(rrf_scores[key], 6),
                # Keep original similarity score if it came from dense
                "retrieval_method": "hybrid"
            }
        ))

    return fused


def _dense_retrieve(
    query: str,
    top_k: int
) -> list[Document]:
    """
    Standard semantic search using BGE embeddings.
    Same as our existing retriever — reproduced here for clarity.
    No similarity threshold here — we want raw ranked candidates
    for RRF to work with, threshold applied after fusion.
    """
    collection = _get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append(Document(
            text=text,
            metadata={
                **metadata,
                "similarity_score": round(1 - distance, 4)
            }
        ))

    return chunks


def _sparse_retrieve(
    query: str,
    top_k: int,
    bm25: BM25Okapi,
    all_texts: list[str],
    all_metas: list[dict]
) -> list[Document]:
    """
    BM25 keyword search over all indexed chunks.
    Scores each chunk by how well it matches query terms,
    weighted by term rarity across the corpus.
    """
    # Tokenize query the same way we tokenized documents
    tokenized_query = query.lower().split()

    # Get BM25 scores for all chunks
    scores = bm25.get_scores(tokenized_query)

    # Get indices of top_k highest scoring chunks
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    chunks = []
    for idx in top_indices:
        # Skip chunks with zero BM25 score
        # (no query terms matched at all)
        if scores[idx] == 0:
            continue

        chunks.append(Document(
            text=all_texts[idx],
            metadata={
                **all_metas[idx],
                "bm25_score": round(float(scores[idx]), 4)
            }
        ))

    return chunks


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    min_rrf_score: float = 0.01,    # RRF scores are small numbers (0.01-0.03)
    dense_candidates: int = 20,      # retrieve more candidates before fusion
    sparse_candidates: int = 20      # then RRF narrows to top_k
) -> list[Document]:
    """
    Full hybrid retrieval:
    1. Dense semantic search (BGE embeddings) → top candidates
    2. Sparse keyword search (BM25) → top candidates
    3. Merge with Reciprocal Rank Fusion
    4. Return top_k fused results

    dense_candidates and sparse_candidates are intentionally larger
    than top_k — we cast a wide net with both methods, then RRF
    picks the best from the combined pool.
    """

    # Build BM25 index from current ChromaDB contents
    bm25, all_texts, all_metas = _build_bm25_index()

    # Run both retrievers in parallel (conceptually —
    # Python runs them sequentially but both complete fast)
    dense_results  = _dense_retrieve(query, top_k=dense_candidates)
    sparse_results = _sparse_retrieve(
        query, top_k=sparse_candidates,
        bm25=bm25, all_texts=all_texts, all_metas=all_metas
    )

    # Fuse results with RRF
    fused = _reciprocal_rank_fusion(dense_results, sparse_results)

    # Apply minimum score threshold and limit to top_k
    filtered = [
        chunk for chunk in fused
        if chunk.metadata.get("rrf_score", 0) >= min_rrf_score
    ][:top_k]

    return filtered


def format_hybrid_chunks(chunks: list[Document]) -> str:
    """Format hybrid results for debugging — shows RRF scores."""
    if not chunks:
        return "No relevant chunks found."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        page  = chunk.metadata.get("page", "?")
        score = chunk.metadata.get("rrf_score", 0)
        sim   = chunk.metadata.get("similarity_score", "n/a")
        bm25  = chunk.metadata.get("bm25_score", "n/a")
        source_label = (
            f"Page {page}" if chunk.metadata.get("type") == "pdf"
            else "Article"
        )
        lines.append(
            f"[Chunk {i} | {source_label} | RRF: {score} "
            f"| Semantic: {sim} | BM25: {bm25}]\n"
            f"{chunk.text}\n"
            f"{'-' * 60}"
        )

    return "\n\n".join(lines)