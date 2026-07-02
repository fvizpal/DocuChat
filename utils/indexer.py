
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from utils.loader import Document


# ---------------------------------------------------------------------------
# Swap to BGE model — better retrieval quality, 768 dimensions
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"  # ← changed

# BGE requires this prefix on QUERIES only (not on documents).
# This is asymmetric embedding — Lesson 4.
# Forgetting this prefix on queries is the #1 BGE mistake.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_embedding_model = None
_chroma_client = None
_collection = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("✅ Embedding model loaded.")
    return _embedding_model


def _get_collection(collection_name: str = "docuchat"):
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_store")
        _collection = _chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _make_chunk_id(chunk: Document) -> str:
    content = (
        f"{chunk.metadata['source']}_"
        f"{chunk.metadata['page']}_"
        f"{chunk.metadata['chunk_index']}_"
        f"{chunk.text[:50]}"
    )
    return hashlib.md5(content.encode()).hexdigest()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed document chunks for indexing.
    NO prefix for documents — BGE is asymmetric.
    Called during indexing only.
    """
    model = _get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True   # ← BGE recommendation: L2 normalize
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a user query for retrieval.
    BGE REQUIRES the query prefix — this is where asymmetric
    embedding from Lesson 4 actually matters in practice.
    Called during retrieval only.
    """
    model = _get_embedding_model()
    prefixed_query = BGE_QUERY_PREFIX + query   # ← prefix on queries only
    embedding = model.encode(
        prefixed_query,
        normalize_embeddings=True   # must match indexing normalization
    )
    return embedding.tolist()


def index_chunks(chunks: list[Document]) -> int:
    """Embed all chunks and store in ChromaDB."""
    if not chunks:
        raise ValueError("No chunks to index.")

    collection = _get_collection()

    texts = [chunk.text for chunk in chunks]

    print(f"Embedding {len(texts)} chunks with BGE model...")
    embeddings = embed_documents(texts)     # ← uses document embedding

    metadatas = [
        {
            "source": chunk.metadata["source"],
            "page": chunk.metadata["page"],
            "type": chunk.metadata["type"],
            "chunk_index": chunk.metadata["chunk_index"],
        }
        for chunk in chunks
    ]

    ids = [_make_chunk_id(chunk) for chunk in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"✅ Indexed {len(chunks)} chunks into ChromaDB.")
    return len(chunks)


def clear_index():
    global _collection
    if _chroma_client is not None:
        _chroma_client.delete_collection("docuchat")
        _collection = None
        print("✅ Index cleared.")


def get_index_stats() -> dict:
    collection = _get_collection()
    return {"total_chunks_indexed": collection.count()}