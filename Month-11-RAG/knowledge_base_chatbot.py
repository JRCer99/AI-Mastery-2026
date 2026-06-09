"""
Personal Knowledge Base Chatbot — Month 11 RAG
Full RAG pipeline: retrieve context → generate grounded answer → chat UI.
Upgrade from Month 10 search: answers questions, doesn't just return snippets.

MOCK = True  → structured mock response (no API needed)
MOCK = False → real LLM via Groq (set GROQ_API_KEY env var)
"""

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os

MOCK = True  # Set False + export GROQ_API_KEY=... when API ready

# ─── Default Notes ───────────────────────────────────────────────────────────

DEFAULT_NOTES = [
    "I am Tyree (J-Astro), CS student at SNHU pursuing AI and software engineering.",
    "2022 Subaru BRZ Limited owner. Currently at ~55k miles, doing fluid maintenance.",
    "Goals 2026: Complete AI Mastery roadmap, strong GitHub portfolio, land an AI role.",
    "Interests: Cars (BRZ, Porsche, GR Supra), cooking Mexican fusion, sim racing, boating.",
    "Health: Managing gastritis — prefer mild foods, avoid spicy, explore gastritis-friendly options.",
    "Current stack: Python, PyTorch, Transformers, LangChain, Git, Docker, Streamlit.",
    "AI Mastery progress: Months 1–10 complete. Month 11 = RAG systems.",
    "Learning style: Project-based with clear milestones and portfolio output.",
    "SNHU courses: CALC 1 (Single-Variable Calculus), DAD-220 (Intro to Databases).",
    "Key projects: Semantic Search Engine, Study With Me Dashboard, AI Code Reviewer, RAG Chatbot.",
]

# ─── Model ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ─── FAISS Index ─────────────────────────────────────────────────────────────

def build_index(notes: list, model):
    embeddings = model.encode(notes).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query: str, notes: list, index, model, top_k: int = 3) -> list:
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, min(top_k, len(notes)))
    return [{"note": notes[i], "score": float(scores[0][j])}
            for j, i in enumerate(indices[0]) if i < len(notes)]

# ─── Generation ──────────────────────────────────────────────────────────────

def generate_answer(query: str, context: list) -> str:
    if MOCK:
        notes_text = "\n".join(f"• {r['note']}" for r in context)
        return (
            f"Based on your personal knowledge base, here's what I found relevant to **\"{query}\"**:\n\n"
            f"{notes_text}\n\n"
            f"*[MOCK mode — plug in Groq or Claude API to generate a synthesized natural language answer.]*"
        )

    # Real LLM path — requires: pip install groq + GROQ_API_KEY env var
    try:
        from groq import Groq
        context_str = "\n".join(f"- {r['note']}" for r in context)
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful personal assistant. Answer using ONLY the context below. "
                    "Be concise and cite which notes support your answer.\n\n"
                    f"Context:\n{context_str}"
                )},
                {"role": "user", "content": query},
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM error] {e}\n\nFalling back to retrieved notes:\n" + "\n".join(f"• {r['note']}" for r in context)

# ─── App ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="wide")

    model = load_model()

    # Session state init
    if "notes" not in st.session_state:
        st.session_state.notes = DEFAULT_NOTES.copy()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index_size" not in st.session_state:
        st.session_state.index_size = 0

    # Rebuild index when notes change
    if st.session_state.index_size != len(st.session_state.notes):
        st.session_state.kb_index = build_index(st.session_state.notes, model)
        st.session_state.index_size = len(st.session_state.notes)

    # ── Sidebar: Note Management ─────────────────────────────────────────────
    with st.sidebar:
        st.title("📝 Knowledge Base")
        api_status = "🟡 MOCK" if MOCK else "🟢 Groq API"
        st.caption(f"{len(st.session_state.notes)} notes indexed | {api_status}")

        new_note = st.text_area("Add note", placeholder="Something to remember...", height=80)
        if st.button("Add", use_container_width=True):
            note = new_note.strip()
            if note and note not in st.session_state.notes:
                st.session_state.notes.append(note)
                st.rerun()

        st.divider()
        for i, note in enumerate(st.session_state.notes):
            c1, c2 = st.columns([9, 1])
            c1.caption(f"{i+1}. {note}")
            if c2.button("✕", key=f"del_{i}"):
                st.session_state.notes.pop(i)
                st.rerun()

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Main: Chat UI ────────────────────────────────────────────────────────
    st.title("🧠 Personal RAG Chatbot")
    st.caption("Ask anything — answers grounded in your knowledge base")

    if MOCK:
        st.info("MOCK mode active. Set `MOCK = False` + `GROQ_API_KEY` for real LLM responses.", icon="🟡")

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "context" in msg:
                with st.expander("📎 Retrieved context"):
                    for hit in msg["context"]:
                        st.markdown(f"**[{hit['score']:.2f}]** {hit['note']}")

    # Chat input
    query = st.chat_input("Ask about yourself...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        context = retrieve(query, st.session_state.notes, st.session_state.kb_index, model)
        answer = generate_answer(query, context)

        st.session_state.messages.append({"role": "assistant", "content": answer, "context": context})
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("📎 Retrieved context"):
                for hit in context:
                    st.markdown(f"**[{hit['score']:.2f}]** {hit['note']}")

    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        for example in ["What are my goals for 2026?", "What's my current tech stack?", "Tell me about my car."]:
            st.markdown(f"- *{example}*")

    st.divider()
    st.caption(f"Month 11 Project 1 — RAG Chatbot | {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
