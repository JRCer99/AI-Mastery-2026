"""
Domain Expert RAG System — Month 11 Project 2
Key upgrade from Project 1: longer documents → chunking → ChromaDB storage.
Demonstrates the full production RAG pattern: chunk → embed → store → retrieve → generate.

MOCK = True  → structured answer (no API needed)
MOCK = False → real LLM via Groq (set GROQ_API_KEY env var)
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os

MOCK = True  # Set False + export GROQ_API_KEY=... when API ready
CHUNK_SIZE = 200   # words per chunk
CHUNK_OVERLAP = 30 # word overlap between chunks

# ─── Domain Corpora ──────────────────────────────────────────────────────────

DOMAINS = {
    "🤖 AI & Machine Learning": [
        """Machine learning is a subset of artificial intelligence where systems learn from data to improve performance
        without being explicitly programmed. There are three main paradigms: supervised learning uses labeled examples
        to train models that predict outputs for new inputs; unsupervised learning finds hidden patterns in unlabeled
        data through clustering and dimensionality reduction; reinforcement learning trains agents to maximize cumulative
        reward through trial-and-error interaction with an environment. Each paradigm suits different problem types.""",

        """Neural networks are computational models inspired by the human brain, consisting of layers of interconnected
        nodes called neurons. A feedforward network passes data from input to output through hidden layers, each applying
        a linear transformation followed by a non-linear activation function like ReLU or sigmoid. Training uses
        backpropagation: compute loss, calculate gradients via the chain rule, and update weights using gradient descent.
        Deep networks with many layers can learn hierarchical representations of complex data like images and text.""",

        """Transformer architecture revolutionized natural language processing. The core mechanism is self-attention,
        which computes relationships between all tokens in a sequence simultaneously. For each token, attention scores
        are computed against all other tokens, allowing the model to focus on relevant context regardless of distance.
        BERT uses bidirectional attention for understanding tasks; GPT uses causal (left-to-right) attention for
        generation. Transformers scale well with data and compute, enabling large language models like GPT-4 and Claude.""",

        """Retrieval-Augmented Generation (RAG) improves LLM accuracy by grounding responses in retrieved documents.
        The pipeline has four stages: chunking splits documents into retrievable pieces; embedding converts chunks to
        dense vectors; retrieval finds the most relevant chunks for a query using similarity search; generation feeds
        retrieved context plus the query to an LLM that synthesizes a grounded answer. RAG reduces hallucination
        because the model cites real source text rather than relying on parametric memory alone.""",

        """Overfitting occurs when a model learns the training data too well, including its noise, and fails to
        generalize to new examples. Signs include high training accuracy but poor validation accuracy. Common
        remedies include: dropout, which randomly deactivates neurons during training to prevent co-adaptation;
        L1/L2 regularization, which penalizes large weights; early stopping, which halts training when validation
        loss stops improving; and data augmentation, which artificially expands the training set with transformed examples.""",
    ],

    "🚗 Automotive Engineering": [
        """The internal combustion engine converts chemical energy from fuel into mechanical work through a four-stroke
        cycle: intake draws the air-fuel mixture into the cylinder; compression raises pressure and temperature;
        combustion ignites the mixture, forcing the piston down and generating torque; exhaust expels burned gases.
        Engine performance is characterized by displacement, compression ratio, bore and stroke, and volumetric
        efficiency. Modern engines use fuel injection, variable valve timing, and turbocharging to maximize power
        while meeting emissions standards.""",

        """Suspension systems maintain tire contact with the road and isolate occupants from disturbances.
        Double-wishbone suspension uses two A-shaped arms to control wheel motion, offering precise geometry control
        favored in performance cars like the Subaru BRZ. MacPherson struts combine a shock absorber and spring into a
        single unit, saving space. Spring rate, damper tuning, anti-roll bar stiffness, and alignment settings
        (camber, caster, toe) all interact to determine handling balance between understeer and oversteer.""",

        """Brake systems convert kinetic energy to heat through friction. Disc brakes clamp brake pads against a
        rotating rotor using hydraulic pressure from the master cylinder. Brake bias — the front-to-rear pressure
        distribution — is critical: too much rear bias causes the rear wheels to lock under hard braking, inducing
        spin. ABS modulates brake pressure to prevent wheel lockup. Performance upgrades include larger rotors for
        more thermal mass, multi-piston calipers for better pad contact, and high-friction compound pads that
        operate well at elevated temperatures.""",
    ],

    "🥗 Nutrition & Health": [
        """Gastritis is inflammation of the stomach lining, often caused by H. pylori bacteria, long-term NSAID use,
        or excessive alcohol. Symptoms include nausea, bloating, upper abdominal pain, and indigestion. Dietary
        management is central to recovery: avoid spicy foods, citrus, caffeine, alcohol, and fried or fatty foods
        that stimulate excess acid. Prefer alkaline or neutral foods such as oatmeal, bananas, lean proteins, steamed
        vegetables, and low-fat dairy. Eating smaller, more frequent meals reduces acid secretion spikes compared
        to large infrequent meals.""",

        """Macronutrients — proteins, carbohydrates, and fats — provide energy and serve distinct structural roles.
        Proteins supply amino acids for muscle synthesis, enzyme production, and immune function; complete proteins
        from animal sources contain all essential amino acids. Carbohydrates are the primary fuel for brain and
        high-intensity exercise; complex carbs from whole grains digest slowly and stabilize blood glucose. Fats
        support hormone synthesis and fat-soluble vitamin absorption; unsaturated fats from olive oil and avocado
        are preferable to saturated and trans fats.""",

        """Hydration affects every physiological process. Water regulates body temperature through sweat, transports
        nutrients and waste, lubricates joints, and maintains blood volume. Dehydration of just 2% of body weight
        impairs cognitive performance and physical endurance. Daily needs vary by size, activity, and climate but
        general guidelines suggest 2–3 liters for most adults. Electrolytes — sodium, potassium, magnesium — must
        be replenished during prolonged exercise. Signs of adequate hydration include pale yellow urine and consistent
        energy levels throughout the day.""",
    ],
}

# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks

def prepare_corpus(docs: list) -> list:
    chunks = []
    for doc_i, doc in enumerate(docs):
        for chunk_i, chunk in enumerate(chunk_text(doc.strip())):
            chunks.append({"id": f"doc{doc_i}_chunk{chunk_i}", "text": chunk})
    return chunks

# ─── ChromaDB ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_chroma(chunks: list, model, collection_name: str):
    client = chromadb.Client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    texts = [c["text"] for c in chunks]
    ids   = [c["id"]   for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=ids)
    return collection

def retrieve(query: str, collection, model, top_k: int = 3) -> list:
    q_emb = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    return [
        {"text": doc, "score": 1 - dist}
        for doc, dist in zip(results["documents"][0], results["distances"][0])
    ]

# ─── Generation ──────────────────────────────────────────────────────────────

def generate_answer(query: str, context: list, domain: str) -> str:
    if MOCK:
        chunks_text = "\n\n".join(f"**[{c['score']:.2f}]** {c['text']}" for c in context)
        return (
            f"**Domain:** {domain}\n\n"
            f"**Retrieved context:**\n\n{chunks_text}\n\n"
            f"*[MOCK mode — set `MOCK = False` + `GROQ_API_KEY` for a synthesized LLM answer.]*"
        )
    try:
        from groq import Groq
        context_str = "\n\n".join(c["text"] for c in context)
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": (
                    f"You are an expert in {domain}. Answer using ONLY the context below. "
                    f"Be precise and cite relevant details.\n\nContext:\n{context_str}"
                )},
                {"role": "user", "content": query},
            ],
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM error] {e}"

# ─── App ─────────────────────────────────────────────────────────────────────

DOMAIN_HINTS = {
    "🤖 AI & Machine Learning": ["How does RAG reduce hallucinations?", "Explain overfitting and how to fix it.", "How does self-attention work?"],
    "🚗 Automotive Engineering": ["How does suspension affect handling?", "What causes brake fade?", "How does the BRZ suspension work?"],
    "🥗 Nutrition & Health":     ["What foods help with gastritis?", "What are macronutrients?", "How much water should I drink daily?"],
}

def main():
    st.set_page_config(page_title="Domain Expert RAG", page_icon="📖", layout="wide")
    model = load_model()

    if "messages"       not in st.session_state: st.session_state.messages = []
    if "active_domain"  not in st.session_state: st.session_state.active_domain = list(DOMAINS.keys())[0]
    if "collection"     not in st.session_state: st.session_state.collection = None

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📖 Domain Expert RAG")
        st.caption("🟡 MOCK mode" if MOCK else "🟢 Groq API")

        selected = st.radio("Select domain", list(DOMAINS.keys()))

        if selected != st.session_state.active_domain or st.session_state.collection is None:
            with st.spinner("Chunking + indexing..."):
                chunks = prepare_corpus(DOMAINS[selected])
                st.session_state.collection = build_chroma(chunks, model, "domain_rag")
                st.session_state.active_domain = selected
                st.session_state.messages = []
            st.success(f"{len(chunks)} chunks indexed")

        st.divider()
        st.caption(f"**Chunk size:** {CHUNK_SIZE} words | **Overlap:** {CHUNK_OVERLAP} words")
        st.caption("Chunking splits long docs into retrievable pieces — core to production RAG.")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Main ─────────────────────────────────────────────────────────────────
    st.title("📖 Domain Expert RAG System")
    st.caption("Long-document RAG: chunk → embed → ChromaDB → retrieve → generate")

    if MOCK:
        st.info("MOCK mode active. Set `MOCK = False` + `GROQ_API_KEY` for real LLM answers.", icon="🟡")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input(f"Ask about {st.session_state.active_domain}...")
    if query and st.session_state.collection:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        context = retrieve(query, st.session_state.collection, model)
        answer  = generate_answer(query, context, st.session_state.active_domain)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        for hint in DOMAIN_HINTS.get(st.session_state.active_domain, []):
            st.markdown(f"- *{hint}*")

    st.divider()
    st.caption(f"Month 11 Project 2 — Domain Expert RAG | {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
