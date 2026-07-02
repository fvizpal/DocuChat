
from utils.loader import load_document, Document
from utils.chunker import chunk_documents
from utils.indexer import index_chunks, clear_index, get_index_stats
from utils.generator import generate_answer


class RAGPipeline:
    """
    Coordinates all RAG components in order:
    load → chunk → index → retrieve → generate

    This is the only class the UI needs to interact with.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 5,
        min_similarity: float = 0.3,
        temperature: float = 0.1
    ):
        # Store config so UI can display/adjust it later
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.temperature = temperature

        # Track what's currently loaded
        self.current_source = None
        self.current_source_type = None
        self.total_chunks = 0
        self.is_ready = False   # False until a document is indexed

    def load(self, source: str, is_url: bool = False) -> dict:
        """
        Full ingestion pipeline: load → chunk → index.
        Call this when the user uploads a PDF or pastes a URL.

        Returns a status dict the UI can use to show feedback.
        """
        try:
            # Step 1: Clear previous document
            # (one document at a time for simplicity)
            clear_index()
            self.is_ready = False

            # Step 2: Load
            print(f"Loading {'URL' if is_url else 'PDF'}: {source}")
            documents = load_document(source, is_url=is_url)

            # Step 3: Chunk
            print(f"Chunking {len(documents)} page(s)...")
            chunks = chunk_documents(
                documents,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            # Step 4: Index
            print(f"Indexing {len(chunks)} chunks...")
            index_chunks(chunks)

            # Update state
            self.current_source = source
            self.current_source_type = "url" if is_url else "pdf"
            self.total_chunks = len(chunks)
            self.is_ready = True

            return {
                "success": True,
                "source": source,
                "source_type": self.current_source_type,
                "pages_loaded": len(documents),
                "chunks_indexed": len(chunks),
                "message": (
                    f"✅ Ready! Indexed {len(chunks)} chunks "
                    f"from {len(documents)} page(s)."
                )
            }

        except Exception as e:
            self.is_ready = False
            return {
                "success": False,
                "message": f"❌ Failed to load document: {str(e)}"
            }

    def ask(self, question: str) -> dict:
        """
        Run the retrieval + generation pipeline for a user question.
        Returns the answer dict from generator.generate_answer().
        """
        if not self.is_ready:
            return {
                "answer": (
                    "No document loaded yet. Please upload a PDF "
                    "or paste a URL first."
                ),
                "sources": [],
                "chunks_used": 0,
                "had_context": False
            }

        if not question.strip():
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "chunks_used": 0,
                "had_context": False
            }

        return generate_answer(
            query=question,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
            temperature=self.temperature
        )

    def status(self) -> dict:
        """Current pipeline state — useful for the UI sidebar."""
        return {
            "is_ready": self.is_ready,
            "current_source": self.current_source,
            "source_type": self.current_source_type,
            "total_chunks": self.total_chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity
        }