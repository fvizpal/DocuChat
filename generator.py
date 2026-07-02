# generator.py
import os
from google import genai
from google.genai import types
from loader import Document
# from retriever import retrieve
from hybrid_retriever import hybrid_retrieve
from reranker import rerank

from query_rewriter import rewrite_query, build_search_queries

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Gemini client — initialized once at module level
# ---------------------------------------------------------------------------

_gemini_client = None

def _get_gemini_client():
    """Initialize Gemini client once and reuse."""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set.\n"
                "Run: export GEMINI_API_KEY='your-key-here'"
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _build_prompt(query: str, chunks: list[Document]) -> str:
    """
    Assemble retrieved chunks into a citation-aware prompt.
    (Lesson 7: metadata + source attribution, grounding instructions
    placed both before AND after context block)
    """

    # Build the context block — each chunk labelled with its source
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        page   = chunk.metadata.get("page", "?")
        doc_type = chunk.metadata.get("type", "unknown")
        score  = chunk.metadata.get("similarity_score", 0)

        # Human-readable source label for citations
        if doc_type == "pdf":
            source_label = f"Page {page} of {source.split('/')[-1]}"
        else:
            source_label = f"Article: {source}"

        context_parts.append(
            f"[Source {i}: {source_label} | Relevance: {score}]\n"
            f"{chunk.text}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    # Lesson 8: grounding instructions before AND after context,
    # explicit "say I don't know" escape hatch,
    # citation format specified clearly
    prompt = f"""You are a helpful assistant that answers questions \
strictly based on the provided context.

INSTRUCTIONS:
- Answer ONLY using information explicitly stated in the context below.
- Do NOT use your general knowledge or training data.
- For every claim you make, cite the source using [Source N] format.
- If the context does not contain enough information to answer the \
question, say: "I don't have enough information in the provided \
document to answer this question."
- Be concise and precise.

CONTEXT:
{context_block}

QUESTION:
{query}

Remember: only use the context above. Cite every claim with [Source N].\
 If the answer isn't there, say so clearly.

ANSWER:"""

    return prompt

# generator.py
# Add these imports at the top:
from query_rewriter import rewrite_query, build_search_queries

# Replace the entire generate_answer() function with this:

def generate_answer(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.3,
    temperature: float = 0.1
) -> dict:
    """
    Full advanced RAG pipeline:
    1. Rewrite query (cleaned + keywords + HyDE)
    2. Multi-query hybrid retrieval across all rewrites
    3. Deduplicate and merge retrieved chunks
    4. Re-rank with cross-encoder
    5. Build grounded prompt with citations
    6. Generate with Gemini
    """
    client = _get_gemini_client()

    # -----------------------------------------------------------------------
    # Stage 1: Query rewriting
    # -----------------------------------------------------------------------
    print(f"\n📝 Rewriting query: '{query}'")
    rewrites = rewrite_query(query)

    print(f"   Cleaned    : {rewrites.get('cleaned_query')}")
    print(f"   Keywords   : {rewrites.get('keywords')}")
    print(f"   Hypothetical: {rewrites.get('hypothetical_answer')}")

    search_queries = build_search_queries(rewrites)

    # -----------------------------------------------------------------------
    # Stage 2: Multi-query hybrid retrieval
    # Run hybrid search for EACH rewritten query, then merge results.
    # This is multi-query retrieval from Lesson 10 — casting a wider net
    # by searching from multiple angles simultaneously.
    # -----------------------------------------------------------------------
    all_candidates: dict[str, Document] = {}
    # key = chunk text (dedup), value = Document

    for search_query in search_queries:
        candidates = hybrid_retrieve(search_query, top_k=top_k * 3)
        for chunk in candidates:
            # Deduplicate by text — same chunk can appear across
            # multiple query results; keep the one with highest RRF score
            key = chunk.text
            existing = all_candidates.get(key)
            if existing is None:
                all_candidates[key] = chunk
            else:
                # Keep whichever retrieval scored it higher
                if (chunk.metadata.get("rrf_score", 0) >
                        existing.metadata.get("rrf_score", 0)):
                    all_candidates[key] = chunk

    merged_candidates = list(all_candidates.values())
    print(f"\n📥 Multi-query retrieved {len(merged_candidates)} "
          f"unique candidates across {len(search_queries)} queries")

    # -----------------------------------------------------------------------
    # Stage 3: Re-rank merged candidates with cross-encoder
    # -----------------------------------------------------------------------
    chunks = rerank(
        query=query,        # use ORIGINAL query for re-ranking
                            # (not the rewrites — we want relevance
                            # to what the user actually asked)
        chunks=merged_candidates,
        top_n=top_k
    )

    print(f"📊 Re-ranked down to {len(chunks)} final chunks")

    # -----------------------------------------------------------------------
    # Stage 4: Handle no-context case
    # -----------------------------------------------------------------------
    if not chunks:
        return {
            "answer": (
                "I couldn't find any relevant information in the "
                "loaded document to answer your question. Try rephrasing "
                "your question, or make sure the right document is loaded."
            ),
            "sources": [],
            "chunks_used": 0,
            "had_context": False,
            "rewrites": rewrites
        }

    # -----------------------------------------------------------------------
    # Stage 5: Build prompt + generate
    # -----------------------------------------------------------------------
    prompt = _build_prompt(query, chunks)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=1024,
        )
    )

    answer = response.text.strip()

    # Collect deduplicated sources
    sources = []
    seen_sources = set()
    for chunk in chunks:
        source   = chunk.metadata.get("source", "unknown")
        page     = chunk.metadata.get("page", "?")
        doc_type = chunk.metadata.get("type", "")
        key      = f"{source}_{page}"

        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({
                "source": source,
                "page": page,
                "type": doc_type,
                "rerank_score": chunk.metadata.get("rerank_score", 0),
                "similarity_score": chunk.metadata.get("similarity_score", 0)
            })

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "had_context": True,
        "rewrites": rewrites      # expose rewrites so UI can show them
    }