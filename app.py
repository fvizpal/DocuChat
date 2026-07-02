
import streamlit as st
from utils.rag_pipeline import RAGPipeline
import tempfile
import os

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocuChat",
    page_icon="📚",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Initialize pipeline in Streamlit session state
# Session state persists across reruns (when user clicks buttons etc.)
# without it, pipeline would reset on every interaction
# ---------------------------------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history

if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False

pipeline = st.session_state.pipeline

# ---------------------------------------------------------------------------
# Sidebar — document loading + pipeline status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 DocuChat")
    st.caption("Chat with any PDF or article")
    st.divider()

    # --- Input type selector ---
    input_type = st.radio(
        "Input type",
        ["📄 PDF Upload", "🌐 URL"],
        horizontal=True
    )

    # --- PDF Upload ---
    if input_type == "📄 PDF Upload":
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            help="Upload any PDF document to chat with it"
        )

        if uploaded_file is not None:
            if st.button("📥 Load PDF", use_container_width=True):
                with st.spinner("Loading and indexing PDF..."):
                    # Save uploaded file to a temp location
                    # (Streamlit gives us a file-like object,
                    # but our loader needs a real file path)
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    result = pipeline.load(tmp_path, is_url=False)
                    os.unlink(tmp_path)  # clean up temp file

                if result["success"]:
                    st.success(result["message"])
                    st.session_state.doc_loaded = True
                    st.session_state.messages = []  # fresh chat
                else:
                    st.error(result["message"])

    # --- URL Input ---
    else:
        url = st.text_input(
            "Paste article URL",
            placeholder="https://example.com/article",
        )

        if st.button("🌐 Load URL", use_container_width=True):
            if url.strip():
                with st.spinner("Fetching and indexing article..."):
                    result = pipeline.load(url.strip(), is_url=True)

                if result["success"]:
                    st.success(result["message"])
                    st.session_state.doc_loaded = True
                    st.session_state.messages = []  # fresh chat
                else:
                    st.error(result["message"])
            else:
                st.warning("Please enter a URL first.")

    # --- Document status ---
    st.divider()
    st.subheader("📊 Document Status")
    status = pipeline.status()

    if status["is_ready"]:
        st.success("✅ Document loaded")
        source = status["current_source"]

        # Truncate long paths/URLs for display
        display_source = (
            source if len(source) <= 50
            else "..." + source[-47:]
        )
        st.caption(f"**Source:** {display_source}")
        st.caption(f"**Type:** {status['source_type'].upper()}")
        st.caption(f"**Chunks indexed:** {status['total_chunks']}")

        # Clear button
        if st.button("🗑️ Clear & Load New", use_container_width=True):
            from utils.indexer import clear_index
            clear_index()
            pipeline.is_ready = False
            st.session_state.doc_loaded = False
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No document loaded yet.")

    # --- Settings expander ---
    st.divider()
    with st.expander("⚙️ Settings"):
        st.caption("Changes apply to the next document loaded.")
        pipeline.top_k = st.slider(
            "Chunks to retrieve (top-k)",
            min_value=1, max_value=10, value=5,
            help="More chunks = more context but slower + pricier"
        )
        pipeline.min_similarity = st.slider(
            "Min similarity threshold",
            min_value=0.0, max_value=1.0, value=0.3, step=0.05,
            help="Higher = stricter matching, fewer but more precise results"
        )
        pipeline.temperature = st.slider(
            "Generation temperature",
            min_value=0.0, max_value=1.0, value=0.1, step=0.05,
            help="Lower = more conservative and grounded answers"
        )

# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------

st.title("💬 Chat with your Document")

# Show welcome message if no document loaded
if not status["is_ready"]:
    st.info(
        "👈 Load a PDF or paste a URL in the sidebar to get started."
    )

    # Show example use cases
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📄 PDFs")
        st.caption(
            "Research papers, contracts, manuals, "
            "reports — any PDF you want to query"
        )
    with col2:
        st.markdown("### 🌐 Articles")
        st.caption(
            "News articles, blog posts, Wikipedia pages "
            "— paste any URL"
        )
    with col3:
        st.markdown("### 💡 Ask Anything")
        st.caption(
            "Ask questions, get summaries, find specific "
            "facts — all with citations"
        )

else:
    # ---------------------------------------------------------------------------
    # Chat history display
    # ---------------------------------------------------------------------------

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources under assistant messages
            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):
                with st.expander(
                    f"📚 Sources ({len(message['sources'])} chunks used)"
                ):
                    for src in message["sources"]:
                        if src["type"] == "pdf":
                            st.caption(
                                f"📄 Page {src['page']} — "
                                f"similarity: {src['similarity_score']}"
                            )
                        else:
                            st.caption(
                                f"🌐 {src['source']} — "
                                f"similarity: {src['similarity_score']}"
                            )
            if (message["role"] == "assistant"
                    and message.get("rewrites")):
                with st.expander("🔍 Query rewrites used"):
                    r = message["rewrites"]
                    st.caption(f"**Original:** {r.get('original_query')}")
                    st.caption(f"**Cleaned:** {r.get('cleaned_query')}")
                    st.caption(f"**Keywords:** {r.get('keywords')}")
                    st.caption(
                        f"**Hypothetical answer:** {r.get('hypothetical_answer')}"
                    )

    # ---------------------------------------------------------------------------
    # Chat input
    # ---------------------------------------------------------------------------

    if prompt := st.chat_input("Ask a question about your document..."):

        # Add user message to history + display it
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.ask(prompt)

            st.markdown(result["answer"])

            # Show sources in expander
            if result["sources"]:
                with st.expander(
                    f"📚 Sources ({result['chunks_used']} chunks used)"
                ):
                    for src in result["sources"]:
                        if src["type"] == "pdf":
                            st.caption(
                                f"📄 Page {src['page']} — "
                                f"similarity: {src['similarity_score']}"
                            )
                        else:
                            st.caption(
                                f"🌐 {src['source']} — "
                                f"similarity: {src['similarity_score']}"
                            )

        # Save assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "rewrites": result.get("rewrites", {}) 
        })