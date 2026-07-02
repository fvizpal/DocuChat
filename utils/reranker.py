
from sentence_transformers import CrossEncoder
from utils.loader import Document


# ---------------------------------------------------------------------------
# Cross-encoder model — loaded once, reused across calls
# Much slower than bi-encoders but far more accurate at judging
# true relevance between a specific query and a specific chunk
# ---------------------------------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def _get_reranker() -> CrossEncoder:
    """Load cross-encoder once and reuse."""
    global _reranker
    if _reranker is None:
        print(f"Loading re-ranker model '{RERANKER_MODEL}'...")
        _reranker = CrossEncoder(
            RERANKER_MODEL,
            max_length=512     # max tokens per query+chunk pair
        )
        print("✅ Re-ranker loaded.")
    return _reranker


def rerank(
    query: str,
    chunks: list[Document],
    top_n: int = 5,
    min_score: float = -5.0    # cross-encoder scores are raw logits
                                # (not 0-1), so threshold is lower
) -> list[Document]:
    """
    Re-rank a list of candidate chunks using a cross-encoder.

    The cross-encoder sees (query, chunk_text) as a PAIR and outputs
    a single relevance score — unlike bi-encoders which score them
    independently. This is why it's more accurate but slower:
    it can only run on a small candidate set, not the whole index.

    Args:
        query   — the user's original question
        chunks  — candidate chunks from hybrid retrieval (typically 20-50)
        top_n   — how many to keep after re-ranking (typically 3-5)
        min_score — discard chunks below this cross-encoder score

    Returns:
        Re-ranked and filtered list of top_n most relevant chunks
    """
    if not chunks:
        return []

    reranker = _get_reranker()

    # Build (query, chunk_text) pairs — this is what the cross-encoder
    # expects. It reads BOTH together, not independently.
    pairs = [(query, chunk.text) for chunk in chunks]

    # Score all pairs — returns a numpy array of raw logit scores
    # Higher = more relevant. Scores are typically in range -10 to +10
    scores = reranker.predict(pairs)

    # Attach score to each chunk and sort descending
    scored_chunks = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )

    # Filter by minimum score and keep top_n
    reranked = []
    for score, chunk in scored_chunks[:top_n]:
        if score < min_score:
            continue

        reranked.append(Document(
            text=chunk.text,
            metadata={
                **chunk.metadata,
                "rerank_score": round(float(score), 4),
                # Keep previous scores for debugging/comparison
                "rrf_score": chunk.metadata.get("rrf_score", "n/a"),
                "similarity_score": chunk.metadata.get(
                    "similarity_score", "n/a"
                ),
            }
        ))

    return reranked


def format_reranked_chunks(chunks: list[Document]) -> str:
    """Format re-ranked results showing all three score layers."""
    if not chunks:
        return "No relevant chunks found after re-ranking."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        page    = chunk.metadata.get("page", "?")
        rerank  = chunk.metadata.get("rerank_score", "?")
        rrf     = chunk.metadata.get("rrf_score", "?")
        sim     = chunk.metadata.get("similarity_score", "?")

        source_label = (
            f"Page {page}" if chunk.metadata.get("type") == "pdf"
            else "Article"
        )

        lines.append(
            f"[Chunk {i} | {source_label}]\n"
            f"  Rerank score : {rerank}  ← cross-encoder (most accurate)\n"
            f"  RRF score    : {rrf}     ← hybrid fusion\n"
            f"  Semantic sim : {sim}     ← BGE cosine\n"
            f"\n{chunk.text}\n"
            f"{'-' * 60}"
        )

    return "\n\n".join(lines)