# Month 10: Vector Embeddings & Vector Databases

**Focus:** Semantic search, embedding models, and vector databases for similarity-based retrieval.

---

## 🎯 Learning Goals

- Generate and store text/image embeddings
- Build semantic search over document collections
- Use vector databases (Pinecone, Chroma, Weaviate)
- Understand cosine similarity and ANN search

---

## 📅 Projects

- [ ] Project 1: Semantic Search Engine over Documents
  - Embed a document corpus, query by meaning not keyword
  - Libraries: `sentence-transformers`, `chromadb` or `pinecone`

- [ ] Project 2: 🏆 Personal Notes → Searchable Knowledge Base *(Weekend Project)*
  - Index personal notes/docs, chat-style search interface
  - Libraries: `sentence-transformers`, `chromadb`, `streamlit`

---

## 🧠 Key Concepts

| Concept | Why It Matters |
|---|---|
| Embeddings | Dense vector representation of meaning |
| Cosine similarity | Measure semantic closeness |
| ANN search | Fast approximate nearest neighbor lookup |
| Chunking | Split docs for optimal retrieval |
| Vector DBs | Purpose-built storage for embeddings |

---

## 📚 Resources

- [Sentence Transformers — Docs](https://www.sbert.net/)
- [ChromaDB — Getting Started](https://docs.trychroma.com/)
- [Pinecone — Vector DB Guide](https://docs.pinecone.io/)

---

## 📂 Structure

```
Month-10-Vector-DBs/
├── semantic_search.py        # Project 1: Semantic Search Engine
└── knowledge_base_app.py     # Project 2: 🏆 Searchable Knowledge Base
```
