"""
Streamlit UI — Autonomous AI Research Agent
Month 12 Capstone · AI Mastery 2026

Two-column layout:
- Left:  live agent activity log (streams as each node completes)
- Right: live report preview (updates when writer/finalizer fire)
"""

import time
import streamlit as st
from knowledge_base import seed_knowledge_base, add_document
from agent import build_graph, AgentState

st.set_page_config(
    page_title="Autonomous AI Research Agent",
    page_icon="🤖",
    layout="wide"
)

# --- sidebar ---

with st.sidebar:
    st.title("🤖 Research Agent")
    st.caption("Month 12 Capstone · AI Mastery 2026")
    st.divider()

    st.subheader("📚 Knowledge Base")
    doc_count = seed_knowledge_base()
    st.success(f"{doc_count} AI/ML documents loaded")

    st.divider()
    st.subheader("➕ Add Document")
    custom_text = st.text_area("Paste text to add:", height=80, key="custom_doc")
    if st.button("Add to KB") and custom_text.strip():
        add_document(f"user_{int(time.time())}", custom_text.strip(), {"source": "user"})
        st.success("Added!")

    st.divider()
    st.subheader("Pipeline")
    st.markdown("""
```
[PLANNER]    break query → sub-questions
    ↓
[RESEARCHER] RAG + web search per sub-Q
    ↓
[WRITER]     synthesize → draft report
    ↓
[CRITIC]     pass/fail + feedback
    ↓  fail → WRITER (max 2 loops)
[FINALIZER]  polish → final report
```
""")
    st.caption("Stack: LangGraph · ChromaDB · Claude API")


# --- main ---

st.title("Autonomous AI Research Agent")
st.caption("Powered by LangGraph + RAG + Claude · AI Mastery 2026 Capstone")

EXAMPLES = [
    "How does RAG reduce hallucination in LLMs?",
    "What makes transformer attention better than RNNs?",
    "How do autonomous AI agents plan and act?",
    "Compare fine-tuning vs prompt engineering for LLMs",
]

query = st.text_input(
    "Research topic or question",
    placeholder="e.g. How do vector databases power semantic search?",
)

ex_cols = st.columns(len(EXAMPLES))
for col, ex in zip(ex_cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state["prefill"] = ex
        st.rerun()

if "prefill" in st.session_state and not query:
    query = st.session_state.pop("prefill")

run_btn = st.button("🚀 Run Research Agent", type="primary", disabled=not query)

if run_btn and query:
    st.divider()
    col_log, col_report = st.columns([1, 2])

    with col_log:
        st.subheader("Agent Activity")
        log_placeholder = st.empty()

    with col_report:
        st.subheader("Research Report")
        report_placeholder = st.empty()
        report_placeholder.info("Waiting for writer node...")

    NODE_ICONS = {
        "planner": "🗺️",
        "researcher": "🔍",
        "writer": "✍️",
        "critic": "🧐",
        "finalizer": "✅",
    }

    all_logs: list[str] = []
    final_report = ""

    graph = build_graph()
    initial: AgentState = {
        "query": query,
        "sub_questions": [],
        "research_notes": "",
        "draft_report": "",
        "critique_passed": False,
        "critique_feedback": "",
        "final_report": "",
        "iteration": 0,
        "log": []
    }

    with st.spinner("Agent running autonomously..."):
        for step in graph.stream(initial):
            for node_name, node_output in step.items():
                for entry in node_output.get("log", []):
                    all_logs.append(entry)
                log_placeholder.markdown(
                    "\n".join(f"- {entry}" for entry in all_logs)
                )

                # show draft as soon as writer fires
                if node_name == "writer" and node_output.get("draft_report"):
                    report_placeholder.markdown(
                        f"*Draft (under critique...)*\n\n{node_output['draft_report']}"
                    )

                # replace with final polished report
                if node_name == "finalizer" and node_output.get("final_report"):
                    final_report = node_output["final_report"]
                    report_placeholder.markdown(final_report)

    if final_report:
        st.success("Research complete!")
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=final_report,
            file_name=f"research_{int(time.time())}.md",
            mime="text/markdown"
        )
    else:
        st.error("Agent did not produce a report. Check that ANTHROPIC_API_KEY is set.")
