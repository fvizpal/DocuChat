# chunker.py
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.loader import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks for embedding.

    chunk_size    — max tokens per chunk (500 is a solid default)
    chunk_overlap — how much text repeats between consecutive chunks
                    (100 = 20% overlap, prevents context loss at boundaries)
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,

        # The splitter tries each separator in order,
        # only moving to the next if the chunk is still too big.
        # This is what makes it "recursive" — it respects structure.
        separators=[
            "\n\n",   # paragraph breaks first (best boundary)
            "\n",     # then line breaks
            ". ",     # then sentence endings
            ", ",     # then clause boundaries
            " ",      # then word boundaries
            ""        # last resort: character-level split
        ],

        # Measure length in characters (simpler and fast;
        # token-based counting needs a tokenizer but is more precise —
        # we can upgrade this later if needed)
        length_function=len,
    )

    all_chunks = []

    for doc in documents:
        # Split this document's text into raw string chunks
        raw_chunks = splitter.split_text(doc.text)

        for i, chunk_text in enumerate(raw_chunks):
            # Skip chunks that are just whitespace
            if not chunk_text.strip():
                continue

            # Each chunk inherits the parent document's metadata
            # PLUS gets its own chunk index for ordering/debugging
            chunk_metadata = {
                **doc.metadata,          # source, page, type from parent
                "chunk_index": i,        # position within this page/doc
                "total_chunks": len(raw_chunks)
            }

            all_chunks.append(Document(
                text=chunk_text,
                metadata=chunk_metadata
            ))

    return all_chunks


def get_chunk_stats(chunks: list[Document]) -> dict:
    """
    Useful for understanding what your chunking produced —
    helpful when debugging retrieval quality later.
    """
    if not chunks:
        return {}

    lengths = [len(c.text) for c in chunks]

    return {
        "total_chunks": len(chunks),
        "avg_length_chars": round(sum(lengths) / len(lengths)),
        "min_length_chars": min(lengths),
        "max_length_chars": max(lengths),
        "sources": list(set(c.metadata["source"] for c in chunks))
    }