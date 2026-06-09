"""
Personal Knowledge Base — Month 10 Project 2 (Weekend Project)
Streamlit app: add notes → FAISS vector index → semantic search UI.
Upgrade from Project 1: FAISS instead of ChromaDB, persistent session state.
"""

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ─── Default Notes ───────────────────────────────────────────────────────────

DEFAULT_NOTES = [
    "I am a CS student at SNHU focusing on AI and software development.",
    "Key skills: Python, PyTorch, Transformers, Pandas, Git, Docker.",
    "Goal for 2026: Complete AI Mastery roadmap and build a strong portfolio.",
    "Preferred learning style: Project-based with clear milestones.",
    "Neural networks learn by adjusting weights through backpropagation.",
    "Transformers use self-attention to process sequences in parallel.",
    "Vector databases store embeddings for fast similarity search.",
    "Cosine similarity measures the angle between two embedding vectors.",
    "Transfer learning reuses pretrained weights to speed up training.",
    "Overfitting occurs when a model memorizes training data too well.",
]


# ─── Model (cached — loads once per session) ─────────────────────────────────

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ─── FAISS Index ─────────────────────────────────────────────────────────────

def build_faiss_index(notes: list, model: SentenceTransformer):
    embeddings = model.encode(notes).astype("float32")
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product = cosine after normalize
    index.add(embeddings)
    return index


def search_notes(query: str, notes: list, index, model: SentenceTransformer, top_k: int = 3):
    query_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, min(top_k, len(notes)))
    return [{"note": notes[i], "score": float(scores[0][j])} for j, i in enumerate(indices[0]) if i < len(notes)]


# ─── App ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Personal Knowledge Base", page_icon="📚", layout="wide")

    st.title("📚 Personal Knowledge Base")
    st.caption("Semantic search over your notes — powered by FAISS + sentence-transformers")

    model = load_model()

    # Init session state
    if "notes" not in st.session_state:
        st.session_state.notes = DEFAULT_NOTES.copy()
    if "index" not in st.session_state or "index_size" not in st.session_state:
        st.session_state.index = build_faiss_index(st.session_state.notes, model)
        st.session_state.index_size = len(st.session_state.notes)

    # Rebuild index if notes changed
    if st.session_state.index_size != len(st.session_state.notes):
        st.session_state.index = build_faiss_index(st.session_state.notes, model)
        st.session_state.index_size = len(st.session_state.notes)

    col_left, col_right = st.columns([1, 1.5])

    # ── Left: Note Management ────────────────────────────────────────────────
    with col_left:
        st.subheader("📝 Notes")
        st.caption(f"{len(st.session_state.notes)} notes indexed")

        new_note = st.text_area("Add a note", placeholder="Type something to remember...", height=80)
        if st.button("Add Note", use_container_width=True):
            note = new_note.strip()
            if note and note not in st.session_state.notes:
                st.session_state.notes.append(note)
                st.success("Note added!")
                st.rerun()
            elif not note:
                st.warning("Note is empty.")
            else:
                st.info("Note already exists.")

        st.divider()
        st.caption("Current notes:")
        for i, note in enumerate(st.session_state.notes):
            col_note, col_del = st.columns([9, 1])
            with col_note:
                st.markdown(f"**{i+1}.** {note}")
            with col_del:
                if st.button("✕", key=f"del_{i}", help="Remove note"):
                    st.session_state.notes.pop(i)
                    st.rerun()

    # ── Right: Search ────────────────────────────────────────────────────────
    with col_right:
        st.subheader("🔍 Semantic Search")
        st.caption("Finds notes by meaning — not just keywords")

        query = st.text_input("Search your knowledge base", placeholder="e.g. what are my goals?")
        top_k = st.slider("Results to show", 1, 5, 3)

        if query.strip():
            results = search_notes(query, st.session_state.notes, st.session_state.index, model, top_k)

            if results:
                st.markdown(f"**Top {len(results)} results for:** *{query}*")
                for hit in results:
                    score_pct = min(hit["score"] * 100, 100)
                    st.markdown(f"""
<div style="background:#1e1e2e;border-left:4px solid #7c3aed;padding:12px 16px;border-radius:6px;margin-bottom:10px">
<div style="font-size:0.8em;color:#a78bfa;margin-bottom:4px">Match: {score_pct:.1f}%</div>
<div>{hit['note']}</div>
</div>
""", unsafe_allow_html=True)
            else:
                st.info("No results found.")

        st.divider()
        st.caption("**How it works:** Notes → embeddings via `all-MiniLM-L6-v2` → FAISS `IndexFlatIP` (cosine similarity after L2 normalize) → top-k nearest neighbors")

    st.divider()
    st.caption(f"Month 10 Project 2 — Personal Knowledge Base | Built {datetime.now().strftime('%B %d, %Y')}")


if __name__ == "__main__":
    main()
