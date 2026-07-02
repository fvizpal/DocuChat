# test_full_pipeline.py
from loader import load_document
from chunker import chunk_documents
from indexer import index_chunks, clear_index
from generator import generate_answer

clear_index()
docs = load_document(
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    is_url=True
)
chunks = chunk_documents(docs)
index_chunks(chunks)

# Test with deliberately messy / colloquial queries
# to see query rewriting in action
queries = [
    "wat is rag and how does it wrk?",         # typos
    "who came up with this retrieval thing?",   # vague/colloquial
    "any downsides to using rag systems?",      # informal phrasing
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"🗣️  Original: {query}")
    result = generate_answer(query)
    print(f"\n💬 Answer:\n{result['answer'][:400]}...")
    print(f"\n🔍 Rewrites used:")
    r = result["rewrites"]
    print(f"   Cleaned    : {r.get('cleaned_query')}")
    print(f"   Keywords   : {r.get('keywords')}")
    print(f"   Hypothetical: {r.get('hypothetical_answer')}")
    print(f"\n📚 Chunks used: {result['chunks_used']}")