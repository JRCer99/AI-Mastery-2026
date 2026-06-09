# Month 09: LLM Orchestration & Agent Foundations

**Focus:** Bridge month — LangChain basics, tool use, and simple multi-step agents. Foundation for Month 12 capstone.

---

## 🎯 Learning Goals

- Understand agent loops: observe → think → act (ReAct pattern)
- Use LangChain for LLM orchestration and tool calling
- Build agents that use tools (time, calculator, search)
- Design modular multi-step reasoning pipelines

---

## 📅 Projects

- [x] Simple Tool-Use Agent ✅
  - Agent with tool calling: time lookup + calculator
  - Libraries: `langchain-core`

- [ ] Multi-tool Agent Skeleton
  - Modular agent framework for Month 12 capstone
  - Libraries: `langchain`, `langgraph`

---

## 🧠 Key Concepts

| Concept | Why It Matters |
|---|---|
| ReAct pattern | Reasoning + Acting loop for agents |
| Tool calling | LLM decides when/how to use tools |
| Memory | Persist context across agent steps |
| Chains | Compose LLM calls into pipelines |
| Agent planning | Break goals into subtasks |

---

## 📚 Resources

- [LangChain — Agents Docs](https://python.langchain.com/docs/modules/agents/)
- [Anthropic — Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use)
- [LangGraph — Getting Started](https://langchain-ai.github.io/langgraph/)

---

## 📂 Structure

```
Month-09-LLM-Orchestration/
├── simple_tool_agent.py    # Project 1: Tool-Use Agent
└── multi_tool_skeleton.py  # Project 2: Multi-tool Skeleton (coming soon)
```
