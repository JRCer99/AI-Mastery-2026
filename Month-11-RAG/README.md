# Month 11: Retrieval-Augmented Generation (RAG)

**Focus:** Build production RAG systems that ground LLM responses in retrieved documents.

---

## 🎯 Learning Goals

- Implement full RAG pipeline: embed → store → retrieve → generate
- Build a conversational interface over a document corpus
- Apply advanced RAG: reranking, hybrid search, query expansion
- Evaluate RAG quality: faithfulness, relevance, answer accuracy

---

## 📅 Projects

- [ ] Project 1: 🏆 Personal Knowledge Base Chatbot — Advanced RAG *(Weekend Project)*
  - Chat with your own notes, docs, or textbooks
  - Libraries: `langchain`, `chromadb`, `streamlit`, `anthropic`

- [ ] Project 2: Domain Expert RAG System
  - RAG over a specialized document set (legal, medical, technical)
  - Libraries: `langchain` or `llama-index`, `pinecone` or `chromadb`

---

## 🧠 Key Concepts

| Concept | Why It Matters |
|---|---|
| RAG pipeline | Retrieve context, then generate grounded answers |
| Reranking | Improve retrieval precision post-search |
| Hybrid search | Combine semantic + keyword retrieval |
| Query expansion | Improve recall with rephrased queries |
| Faithfulness | LLM answers grounded in retrieved docs |

---

## 📚 Resources

- [LangChain — RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex — RAG Guide](https://docs.llamaindex.ai/en/stable/understanding/rag/)
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

---

## 📂 Structure

```
Month-11-RAG/
├── knowledge_base_chatbot.py   # Project 1: 🏆 KB Chatbot (Advanced RAG)
└── domain_expert_rag.py        # Project 2: Domain Expert RAG
```
