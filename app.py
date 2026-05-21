# ============================================================
# app.py — Streamlit RAG Chatbot — Production-Ready
# ============================================================

import os
import shutil
import time
import traceback
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import (
    DATA_PATH,
    EMBEDDINGS_PATH,
    USE_LOCAL_EMBEDDINGS,
    RETRIEVAL_K,
    NUM_SUGGESTIONS,
)


# ──────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ──────────────────────────────────────────────────────────────────────

def _init_session_state():
    defaults = {
        "messages": [],
        "qa_chain": None,
        "chunks": [],
        "suggestions": [],
        "last_context": None,
        "use_hybrid": True,
        "use_summarization": True,
        "show_sources": True,
        "semantic_weight": 0.6,
        "pending_question": "",
        "doc_count": 0,
        "chunk_count": 0,
        "chain_ready": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session_state()


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _embeddings_exist() -> bool:
    idx_file = Path(EMBEDDINGS_PATH) / "index.faiss"
    return idx_file.exists()


def _save_uploaded_file(uploaded_file) -> Path:
    """Save a Streamlit UploadedFile to the appropriate data/ subfolder."""
    ext_to_dir = {
        "pdf": "pdfs",
        "txt": "txt",
        "md": "md",
        "docx": "docx",
        "html": "html",
        "json": "json",
    }
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    target_dir = Path(DATA_PATH) / ext_to_dir.get(ext, "txt")
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getvalue())
    return dest


def _run_ingestion_pipeline(progress_bar=None) -> tuple[list, object]:
    """Run ingest → embed → save, return (chunks, vector_store)."""
    from src.ingest import load_documents, clean_documents, chunk_documents
    from src.embeddings import get_embedding_model, create_vector_store, save_vector_store

    if progress_bar:
        progress_bar.progress(10, text="Loading documents …")

    docs = load_documents(DATA_PATH)
    cleaned = clean_documents(docs)

    if progress_bar:
        progress_bar.progress(40, text="Chunking documents …")

    chunks = chunk_documents(cleaned)

    if progress_bar:
        progress_bar.progress(60, text="Building embeddings …")

    embed_model = get_embedding_model(use_local=USE_LOCAL_EMBEDDINGS)
    vector_store = create_vector_store(chunks, embed_model)
    save_vector_store(vector_store, EMBEDDINGS_PATH)

    if progress_bar:
        progress_bar.progress(100, text="Done!")

    return chunks, vector_store


def _build_chain(chunks=None):
    """Create (or recreate) the QA chain from current session settings."""
    from src.chain import create_qa_chain

    chain = create_qa_chain(
        use_hybrid=st.session_state.use_hybrid,
        use_summarization=st.session_state.use_summarization,
        chunks=chunks or st.session_state.chunks,
    )
    return chain


# ──────────────────────────────────────────────────────────────────────
# Auto-load chain on first run (if embeddings exist)
# ──────────────────────────────────────────────────────────────────────

if not st.session_state.chain_ready and _embeddings_exist():
    with st.spinner("Loading existing index …"):
        try:
            st.session_state.qa_chain = _build_chain()
            st.session_state.chain_ready = True
        except Exception as e:
            st.warning(f"Could not auto-load chain: {e}")


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 RAG Chatbot Settings")

    # ── Document upload ──────────────────────────────────────────────
    st.subheader("📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "docx", "html", "json"],
        accept_multiple_files=True,
        help="Drop files here or click to browse. Supported: PDF, TXT, MD, DOCX, HTML, JSON.",
    )

    if st.button("⚙️ Process New Documents", use_container_width=True):
        if uploaded_files:
            progress = st.progress(0, text="Saving files …")
            for f in uploaded_files:
                dest = _save_uploaded_file(f)
                progress.progress(5, text=f"Saved {f.name}")
        else:
            progress = st.progress(0, text="Re-indexing existing documents …")

        try:
            chunks, _ = _run_ingestion_pipeline(progress)
            st.session_state.chunks = chunks
            st.session_state.chunk_count = len(chunks)
            st.session_state.qa_chain = _build_chain(chunks)
            st.session_state.chain_ready = True
            st.success(f"✅ Indexed {len(chunks)} chunks. Ready to chat!")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            st.code(traceback.format_exc())

    st.divider()

    # ── Enhancement toggles ──────────────────────────────────────────
    st.subheader("⚙️ Enhancements")

    new_hybrid = st.checkbox(
        "Hybrid Search (BM25 + Semantic)",
        value=st.session_state.use_hybrid,
        help="Combine keyword and semantic retrieval for better coverage.",
    )
    new_summarization = st.checkbox(
        "Conversation Summarization",
        value=st.session_state.use_summarization,
        help="Automatically compress long conversations to save tokens.",
    )
    show_sources = st.checkbox(
        "Show Source Citations",
        value=st.session_state.show_sources,
        help="Display the source documents used for each answer.",
    )

    st.session_state.show_sources = show_sources

    semantic_weight = st.slider(
        "Hybrid: Semantic Weight",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.semantic_weight,
        step=0.05,
        help="1.0 = purely semantic; 0.0 = purely keyword.",
    )
    st.session_state.semantic_weight = semantic_weight

    # Rebuild chain if key settings changed
    settings_changed = (
        new_hybrid != st.session_state.use_hybrid
        or new_summarization != st.session_state.use_summarization
    )
    if settings_changed:
        st.session_state.use_hybrid = new_hybrid
        st.session_state.use_summarization = new_summarization
        if st.session_state.chain_ready:
            with st.spinner("Rebuilding chain …"):
                try:
                    st.session_state.qa_chain = _build_chain()
                    st.success("Chain updated.")
                except Exception as e:
                    st.error(f"Could not rebuild chain: {e}")

    st.divider()

    # ── Memory management ────────────────────────────────────────────
    st.subheader("🧠 Memory Management")

    if st.session_state.chain_ready and st.session_state.use_summarization:
        try:
            from src.memory_manager import SummarizingMemory
            mem = st.session_state.qa_chain.memory
            if isinstance(mem, SummarizingMemory):
                token_count = mem.get_total_tokens()
                st.metric("Current token count", f"{token_count:,}")
                if st.button("🗜 Compress History Now", use_container_width=True):
                    mem.prune_memory()
                    st.success("Conversation compressed.")
        except Exception:
            pass

    if st.button("🗑 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.suggestions = []
        st.session_state.last_context = None
        if st.session_state.qa_chain:
            try:
                st.session_state.qa_chain.memory.clear()
            except Exception:
                pass
        st.rerun()

    st.divider()

    # ── Statistics ───────────────────────────────────────────────────
    st.subheader("📊 Statistics")
    st.write(f"**Chunks indexed:** {st.session_state.chunk_count or '—'}")
    st.write(f"**Embedding model:** {'Local (MiniLM)' if USE_LOCAL_EMBEDDINGS else 'OpenAI'}")
    st.write(f"**Retrieval k:** {RETRIEVAL_K}")
    st.write(f"**Hybrid search:** {'On' if st.session_state.use_hybrid else 'Off'}")
    st.write(f"**Summarization:** {'On' if st.session_state.use_summarization else 'Off'}")


# ──────────────────────────────────────────────────────────────────────
# MAIN CHAT INTERFACE
# ──────────────────────────────────────────────────────────────────────

st.title("💬 Ask Questions About Your Documents")

if st.session_state.chain_ready:
    st.caption("✅ Documents are indexed and ready for questions.")
else:
    st.info(
        "👆 Upload documents and click **Process New Documents** in the sidebar to get started, "
        "or place files in the `data/` subfolders and process them."
    )

st.divider()

# ── Render chat history ──────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant" and st.session_state.show_sources:
            sources = msg.get("sources", [])
            if sources:
                with st.expander("📚 View Sources"):
                    for i, src in enumerate(sources, start=1):
                        filename = src.metadata.get("filename", src.metadata.get("source", "Unknown"))
                        page = src.metadata.get("page", "")
                        score = src.metadata.get("hybrid_score", src.metadata.get("relevance_score", ""))
                        st.markdown(f"**Source {i}: `{filename}`**" + (f" — page {page}" if page else ""))
                        if score:
                            st.caption(f"Relevance score: {score:.4f}" if isinstance(score, float) else f"Score: {score}")
                        st.text(src.page_content[:300] + ("…" if len(src.page_content) > 300 else ""))
                        if i < len(sources):
                            st.markdown("---")

# ── Suggestion buttons (displayed between history and input) ─────────

if st.session_state.suggestions:
    st.caption("💡 You might also ask:")
    cols = st.columns(len(st.session_state.suggestions))
    for col, suggestion in zip(cols, st.session_state.suggestions):
        with col:
            if st.button(suggestion, use_container_width=True, key=f"sug_{suggestion[:30]}"):
                st.session_state.pending_question = suggestion
                st.rerun()

# ── Chat input ───────────────────────────────────────────────────────

# Allow suggestion buttons to pre-fill the input
if st.session_state.pending_question:
    prefill = st.session_state.pending_question
    st.session_state.pending_question = ""
    prompt = prefill
else:
    prompt = st.chat_input(
        "Ask a question about your documents …",
        disabled=not st.session_state.chain_ready,
    )

# ── Process prompt ───────────────────────────────────────────────────

if prompt:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()

        try:
            with st.spinner("Thinking …"):
                result = st.session_state.qa_chain.invoke({"question": prompt})

            answer = result.get("answer", "I could not generate an answer.")
            source_docs = result.get("source_documents", [])

            # Simulate streaming by writing word by word
            displayed = ""
            for word in answer.split():
                displayed += word + " "
                answer_placeholder.markdown(displayed + "▌")
                time.sleep(0.01)
            answer_placeholder.markdown(answer)

            # Source citations
            if source_docs and st.session_state.show_sources:
                with st.expander("📚 View Sources"):
                    for i, src in enumerate(source_docs, start=1):
                        filename = src.metadata.get("filename", src.metadata.get("source", "Unknown"))
                        page = src.metadata.get("page", "")
                        score = src.metadata.get("hybrid_score", src.metadata.get("relevance_score", ""))
                        st.markdown(f"**Source {i}: `{filename}`**" + (f" — page {page}" if page else ""))
                        if score:
                            st.caption(f"Score: {score:.4f}" if isinstance(score, float) else f"Score: {score}")
                        st.text(src.page_content[:300] + ("…" if len(src.page_content) > 300 else ""))
                        if i < len(source_docs):
                            st.markdown("---")

            # Persist message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_docs,
            })

            # Store context for suggestion generation
            st.session_state.last_context = source_docs

            # Generate follow-up suggestions asynchronously
            if source_docs:
                try:
                    from src.chain import generate_similar_questions
                    llm = st.session_state.qa_chain.combine_docs_chain.llm_chain.llm
                    suggestions = generate_similar_questions(source_docs, prompt, llm, NUM_SUGGESTIONS)
                    st.session_state.suggestions = suggestions
                except Exception as e:
                    print(f"[WARNING] Suggestion generation failed: {e}")
                    st.session_state.suggestions = []

        except Exception as e:
            error_msg = f"⚠️ An error occurred: {str(e)}"
            answer_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
            })
            print(f"[ERROR] {traceback.format_exc()}")

    # Rerun to render suggestions below chat history
    if st.session_state.suggestions:
        st.rerun()