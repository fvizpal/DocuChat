
from indexer import embed_query, _get_collection   # ← embed_query instead
from loader import Document


def retrieve(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.3
) -> list[Document]:
    """
    Embed query with BGE prefix and search ChromaDB.
    """
    collection = _get_collection()

    if collection.count() == 0:
        raise ValueError(
            "No documents indexed yet. "
            "Please load and index a document first."
        )

    # Use embed_query (with BGE prefix) — NOT embed_documents
    # This is the asymmetric embedding distinction from Lesson 4
    query_embedding = embed_query(query)    # ← key change here

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_chunks = []

    for text, metadata, distance in zip(docs, metadatas, distances):
        similarity = 1 - distance

        if similarity < min_similarity:
            continue

        retrieved_chunks.append(Document(
            text=text,
            metadata={
                **metadata,
                "similarity_score": round(similarity, 4)
            }
        ))

    return retrieved_chunks


def format_retrieved_chunks(chunks: list[Document]) -> str:
    if not chunks:
        return "No relevant chunks found."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        page  = chunk.metadata.get("page", "?")
        score = chunk.metadata.get("similarity_score", 0)
        source_label = (
            f"Page {page}" if chunk.metadata.get("type") == "pdf"
            else "Article"
        )
        lines.append(
            f"[Chunk {i} | {source_label} | Score: {score}]\n"
            f"{chunk.text}\n"
            f"{'-' * 60}"
        )

    return "\n\n".join(lines)