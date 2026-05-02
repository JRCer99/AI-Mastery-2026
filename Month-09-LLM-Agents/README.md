# Month 09: LLM Orchestration & Agent Foundations

**Focus:** Build multi-step reasoning agents using LangChain/LlamaIndex and tool-use patterns.

---

## 🎯 Learning Goals

- Understand agent loops: observe → think → act
- Use LangChain and LlamaIndex for LLM orchestration
- Build agents that use tools (search, calculator, code exec)
- Design multi-step reasoning pipelines

---

## 📅 Projects

- [ ] LangChain / LlamaIndex Basics
  - Chains, prompts, memory, retrievers
  - Libraries: `langchain`, `llama-index`

- [ ] Simple Tool-Use Agent
  - Agent with 2–3 tools (search, calculator, file reader)
  - Libraries: `langchain`, `anthropic` or `openai`

- [ ] Multi-tool Agent Skeleton
  - Modular agent framework for Month 12 capstone
  - Libraries: `langchain` or `langgraph`

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
- [LlamaIndex — Getting Started](https://docs.llamaindex.ai/en/stable/)
- [Anthropic — Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use)

---

## 📂 Structure

```
Month-09-LLM-Agents/
├── langchain_basics.py     # LangChain / LlamaIndex intro
├── tool_use_agent.py       # Simple Tool-Use Agent
└── multi_tool_skeleton.py  # Multi-tool Agent Skeleton
```
