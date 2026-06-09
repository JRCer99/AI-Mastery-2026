# Autonomous AI Research Agent

> Month 12 Capstone · AI Mastery 2026

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)
![Claude](https://img.shields.io/badge/Claude-Sonnet%204.6-blueviolet?logo=anthropic)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)

A fully autonomous AI research agent that takes any question, plans a research strategy, retrieves relevant knowledge via RAG, writes a structured report, critiques its own output, revises if needed, and delivers a polished markdown report — all without human involvement.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│   PLANNER   │  Breaks query into 3-4 focused sub-questions
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  RESEARCHER │  Per sub-question: ChromaDB RAG + web search
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   WRITER    │◄─────────────────────┐
└──────┬──────┘                      │ (revision loop,
       │                             │  max 2 iterations)
       ▼                             │
┌─────────────┐  FAIL + iter < 2    │
│   CRITIC    │─────────────────────┘
└──────┬──────┘
       │ PASS or iter >= 2
       ▼
┌─────────────┐
│  FINALIZER  │  Polishes report to publication quality
└──────┬──────┘
       │
       ▼
 Markdown Report + Download
```

State machine built with **LangGraph** — each node is a pure function operating on shared `AgentState`. The critic's conditional edge creates an autonomous revision loop with a safety cap of 2 iterations.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp env.example .env
# Edit .env — add: ANTHROPIC_API_KEY=sk-ant-...

# 3. Run
streamlit run app.py
```

ChromaDB auto-seeds 15 AI/ML knowledge base documents on first run. No additional setup needed.

---

## Usage

1. Enter any research question in the text field
2. Click **Run Research Agent**
3. Watch the live activity log as each node executes
4. Read and download the final polished report

**Example queries:**
- How does RAG reduce hallucination in LLMs?
- What makes transformer attention better than RNNs?
- Compare fine-tuning vs prompt engineering for LLMs
- How do autonomous AI agents plan and act?

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | LangGraph (StateGraph) |
| LLM — planning, writing, critique | Claude Sonnet 4.6 |
| LLM — web search simulation | Claude Haiku 4.5 |
| Vector store / RAG | ChromaDB + ONNX embeddings |
| UI | Streamlit |
| Env management | python-dotenv |

---

## Skills Demonstrated

This capstone integrates skills from all 12 months:

| Month | Skill Applied |
|-------|--------------|
| 1 | Python architecture, clean code |
| 7 | LLM prompting, Claude API |
| 8 | MLOps patterns, environment management |
| 9 | Agent orchestration, tool use |
| 10–11 | Vector embeddings, RAG pipeline |
| 12 | Multi-agent systems, LangGraph state machines |

---

## Files

```
capstone_agent/
├── agent.py          # LangGraph state machine (5 nodes, conditional edges)
├── tools.py          # RAG search + mock web search tools
├── knowledge_base.py # ChromaDB setup + 15 seed documents
├── app.py            # Streamlit UI with live streaming
├── requirements.txt  # Dependencies
└── env.example       # API key template
```
