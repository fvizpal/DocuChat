# DocuChat

Chat with PDFs and web articles using retrieval-augmented generation (RAG). Upload a document or paste a URL, then ask questions — answers include source citations.

## Features

- **PDF & URL support** — upload any PDF or paste any article URL
- **Hybrid retrieval** — dense (BGE embeddings) + sparse (BM25) search fused via Reciprocal Rank Fusion
- **Query rewriting** — Gemini-powered query cleaning, keyword extraction, and HyDE (Hypothetical Document Embedding)
- **Multi-query retrieval** — search from multiple angles for broader coverage
- **Cross-encoder re-ranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` pinpoints the most relevant chunks
- **Source citations** — every answer cites its sources with page numbers or URLs
- **Streamlit UI** — clean chat interface with settings controls

## Architecture

```
app.py                  Streamlit UI (entry point)
utils/
├── loader.py           PDF (PyMuPDF) & URL (trafilatura) loading
├── chunker.py          Recursive text splitting
├── indexer.py          BGE embeddings + ChromaDB vector store
├── retriever.py        Semantic search (dense retrieval)
├── hybrid_retriever.py Dense + BM25 hybrid with RRF fusion
├── reranker.py         Cross-encoder re-ranking
├── query_rewriter.py   Gemini-powered query rewriting
├── generator.py        Prompt building + Gemini generation
└── rag_pipeline.py     Orchestrates all components
```

## Requirements

- Python 3.12+
- Gemini API key (set in `.env`)

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd DocuChat

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-key-here
```

## Usage

```bash
streamlit run app.py
```

Open the URL printed in the terminal (default: `http://localhost:8501`). Upload a PDF or paste a URL in the sidebar, then start asking questions.

## Project structure notes

The project currently lacks a few standard conventions that would improve maintainability:

- **`requirements.txt`** — dependencies are not pinned; users must install manually from their venv
- **Package layout** — source files live in `utils/` without `__init__.py` files; a proper `src/docuchat/` package with `pyproject.toml` would enable `pip install -e .`
- **Tests** — `test_*.py` files sit at the project root; a `tests/` directory is conventional
- **Configuration** — settings are spread across modules (model names, chunk sizes, paths); a central config class or YAML file would be cleaner

These are incremental improvements, not blockers — the code works as-is.
