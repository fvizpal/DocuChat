# query_rewriter.py
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ---------------------------------------------------------------------------
# We reuse the same Gemini client pattern from generator.py
# ---------------------------------------------------------------------------

_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def rewrite_query(query: str) -> dict:
    """
    Use Gemini to rewrite the user's query in three complementary ways:
    
    1. Cleaned query    — fix typos, expand abbreviations, clarify intent
    2. Keywords         — extract key terms for BM25 keyword search
    3. Hypothetical doc — a short hypothetical answer (HyDE from Lesson 10)
                          whose vocabulary matches how documents are written

    Returns all three so the retriever can use whichever fits best,
    or combine them for maximum coverage.
    """
    client = _get_gemini_client()

    # We ask for all three rewrites in one API call (cheaper + faster)
    # and request JSON output so it's easy to parse
    prompt = f"""You are a search query optimizer for a RAG system.

Given a user's question, produce THREE search-optimized versions.
Respond ONLY with valid JSON, no explanation, no markdown backticks.

{{
  "cleaned_query": "rewritten version of the question with typos fixed,
                    abbreviations expanded, and intent made explicit",
  "keywords": "3-6 key search terms extracted from the question,
               space-separated, focused on nouns and technical terms",
  "hypothetical_answer": "a single sentence that looks like how a document
                          would state the answer to this question, using
                          formal vocabulary a document might use"
}}

User question: {query}

JSON response:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,        # low temp for consistent rewrites
            max_output_tokens=256,  # rewrites should be short
        )
    )

    raw = response.text.strip()

    # Strip markdown code fences if Gemini adds them despite instructions
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    try:
        import json
        rewrites = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: if JSON parsing fails, use original query for everything
        print(f"⚠️ Query rewrite JSON parse failed, using original query.")
        rewrites = {
            "cleaned_query": query,
            "keywords": query,
            "hypothetical_answer": query
        }

    # Always include the original query too
    rewrites["original_query"] = query
    return rewrites


def build_search_queries(rewrites: dict) -> list[str]:
    """
    From the rewrite dict, build a list of distinct search queries
    to run through hybrid retrieval.

    We use:
    - cleaned_query     → main semantic search (best intent capture)
    - keywords          → good for BM25 term matching
    - hypothetical_answer → HyDE: matches document vocabulary (Lesson 10)

    Deduplicating ensures we don't run the same query twice
    if rewrites are similar to each other.
    """
    seen = set()
    queries = []

    for key in ["cleaned_query", "keywords", "hypothetical_answer"]:
        q = rewrites.get(key, "").strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)

    # Always have at least the original as fallback
    if not queries:
        queries = [rewrites.get("original_query", "")]

    return queries