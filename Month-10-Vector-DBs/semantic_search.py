"""
Semantic Search Engine — Month 10 Vector Embeddings & Vector Databases
Embeds a document corpus with sentence-transformers, stores in ChromaDB,
and retrieves by meaning — not keyword matching.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ─── Corpus ──────────────────────────────────────────────────────────────────

DOCUMENTS = [
    {"id": "doc1",  "text": "Neural networks learn by adjusting weights through backpropagation and gradient descent."},
    {"id": "doc2",  "text": "Convolutional neural networks excel at image recognition by detecting spatial patterns."},
    {"id": "doc3",  "text": "Transformers use self-attention to process sequences in parallel, enabling faster training."},
    {"id": "doc4",  "text": "Reinforcement learning trains agents to maximize rewards through trial and error interaction."},
    {"id": "doc5",  "text": "Vector databases store embeddings and enable fast similarity search over large datasets."},
    {"id": "doc6",  "text": "Transfer learning reuses pretrained model weights to speed up training on new tasks."},
    {"id": "doc7",  "text": "BERT is a bidirectional transformer pretrained on masked language modeling and next sentence prediction."},
    {"id": "doc8",  "text": "Cosine similarity measures the angle between two vectors — used to compare embedding closeness."},
    {"id": "doc9",  "text": "Data augmentation artificially expands training sets to improve model generalization."},
    {"id": "doc10", "text": "Overfitting occurs when a model memorizes training data and fails to generalize to new examples."},
    {"id": "doc11", "text": "Recurrent neural networks process sequential data by maintaining a hidden state across time steps."},
    {"id": "doc12", "text": "Generative adversarial networks pit a generator against a discriminator to produce realistic outputs."},
    {"id": "doc13", "text": "Tokenization splits raw text into subword units that transformer models can process numerically."},
    {"id": "doc14", "text": "Regularization techniques like dropout and L2 penalize model complexity to reduce overfitting."},
    {"id": "doc15", "text": "Semantic search retrieves documents based on meaning rather than exact keyword overlap."},
]


# ─── Setup ───────────────────────────────────────────────────────────────────

def build_index(model: SentenceTransformer) -> chromadb.Collection:
    """Embed corpus and load into ChromaDB in-memory collection."""
    client = chromadb.Client()  # in-memory — no server needed
    collection = client.get_or_create_collection(
        name="ai_knowledge_base",
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    texts = [d["text"] for d in DOCUMENTS]
    ids   = [d["id"]   for d in DOCUMENTS]

    print("Encoding documents...", end=" ", flush=True)
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    print("done.")

    collection.add(documents=texts, embeddings=embeddings, ids=ids)
    return collection


# ─── Search ──────────────────────────────────────────────────────────────────

def search(query: str, collection: chromadb.Collection, model: SentenceTransformer, top_k: int = 3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    hits = []
    for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
        similarity = 1 - dist  # ChromaDB cosine returns distance, not similarity
        hits.append({"rank": i + 1, "text": doc, "score": similarity})
    return hits


# ─── Demo ────────────────────────────────────────────────────────────────────

def run_demo(collection, model):
    demo_queries = [
        "How do machines learn from data?",
        "Finding similar items quickly in large datasets",
        "Preventing a model from memorizing training examples",
    ]

    print("\n── Demo Queries ──")
    for query in demo_queries:
        print(f"\nQuery: \"{query}\"")
        for hit in search(query, collection, model):
            print(f"  #{hit['rank']} [{hit['score']:.3f}] {hit['text']}")


def run_interactive(collection, model):
    print("\n── Interactive Search (type 'quit' to exit) ──")
    while True:
        query = input("\nSearch: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        for hit in search(query, collection, model, top_k=3):
            print(f"  #{hit['rank']} [{hit['score']:.3f}] {hit['text']}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Semantic Search Engine — Month 10 Vector Embeddings & Vector DBs")
    print("=" * 65)
    print(f"Model:  all-MiniLM-L6-v2  |  Corpus: {len(DOCUMENTS)} docs  |  DB: ChromaDB (in-memory)")
    print("First run downloads ~80MB model — cached after that.\n")

    print("Loading model...", end=" ", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("done.")

    collection = build_index(model)
    print(f"Index built — {collection.count()} vectors stored.")

    run_demo(collection, model)
    run_interactive(collection, model)

    print(f"\n🎉 Month 10 Project 1 Complete! — {datetime.now().strftime('%B %d, %Y')}")
    print("Foundation set — Project 2 adds a Streamlit UI over this same engine.")

if __name__ == "__main__":
    main()
