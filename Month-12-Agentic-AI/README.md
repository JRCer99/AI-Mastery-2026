# Month 12: Agentic AI & Autonomous Multimodal Agents

**Focus:** Build fully autonomous multi-agent systems and ship a capstone project showcasing the full 12-month journey.

---

## 🎯 Learning Goals

- Design multi-agent architectures (CrewAI / LangGraph)
- Build agents that handle multimodal inputs (text + image)
- Ship a browser extension powered by AI
- Deliver a capstone autonomous agent system

---

## 📅 Projects

- [x] Project 1: Multi-Agent Research Team ✅
  - CrewAI-compatible Agent/Task/Crew pipeline — Researcher → Writer, saves markdown report
  - Libraries: pure Python (CrewAI interface built from scratch for Python 3.9 compat)

- [x] Project 2: Autonomous Multimodal Assistant ✅
  - Upload image + chat — Claude Vision with MOCK=True flag, side-by-side layout
  - Libraries: `streamlit`, `Pillow`, `anthropic` (vision, live mode)

- [x] Project 3: 🏆 Chrome Extension — AI Summarizer ✅ *(Weekend Project)*
  - Manifest V3, page text extraction, Claude API summarization, MOCK=true flag
  - Stack: JavaScript, Chrome Extensions API, Anthropic API

- [ ] Capstone: Fully Autonomous AI Agent System
  - End-to-end autonomous agent using all skills from Months 1–12
  - Stack: Python, LangGraph, RAG, vector DB, Claude API

---

## 🧠 Key Concepts

| Concept | Why It Matters |
|---|---|
| Multi-agent systems | Parallel specialized agents outperform solo LLMs |
| LangGraph | State machine for complex agent workflows |
| Multimodal inputs | Process text + images in one pipeline |
| Chrome Extensions | Ship AI directly into the browser |
| Autonomous loops | Agent operates without human in the loop |

---

## 📚 Resources

- [CrewAI — Docs](https://docs.crewai.com/)
- [LangGraph — Getting Started](https://langchain-ai.github.io/langgraph/)
- [Chrome Extensions — Official Guide](https://developer.chrome.com/docs/extensions/)

---

## 📂 Structure

```
Month-12-Agentic-AI/
├── multi_agent_research.py     # Project 1: Multi-Agent Team
├── multimodal_assistant.py     # Project 2: Multimodal Agent
├── chrome_extension/           # Project 3: 🏆 Chrome Extension
└── capstone_agent/             # Capstone: Autonomous Agent System
```
