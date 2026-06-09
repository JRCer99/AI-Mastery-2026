"""
ChromaDB knowledge base — seeded with 15 AI/ML documents from Months 1-12.
Provides semantic search for the RAG pipeline.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

SEED_DOCS = [
    ("kb_01", "Neural networks are composed of layered nodes that learn by adjusting weights via backpropagation. Deep learning stacks many layers to capture hierarchical features. CNNs use convolutional filters for spatial patterns in images. RNNs and LSTMs handle sequential data with memory.", {"topic": "deep_learning"}),
    ("kb_02", "Transformers use self-attention to process all tokens in parallel, overcoming RNN bottlenecks. BERT uses bidirectional attention for understanding tasks. GPT uses causal (left-to-right) attention for generation. The attention formula is: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V.", {"topic": "transformers"}),
    ("kb_03", "Large Language Models are trained via next-token prediction on massive corpora. RLHF aligns model outputs with human preferences. Prompt engineering techniques include chain-of-thought, few-shot, zero-shot, and system prompts.", {"topic": "llms"}),
    ("kb_04", "RAG (Retrieval-Augmented Generation) mitigates hallucination by grounding LLM responses in retrieved documents. Steps: chunk documents, embed chunks, store in vector DB, retrieve top-k on query, inject into prompt. Reduces need for fine-tuning for knowledge-intensive tasks.", {"topic": "rag"}),
    ("kb_05", "Vector databases store dense embeddings for semantic similarity search. ChromaDB, Pinecone, Weaviate, and Qdrant are leading options. HNSW indexing enables fast approximate nearest-neighbor search. Cosine similarity and dot product are common distance metrics.", {"topic": "vector_databases"}),
    ("kb_06", "AI agents combine LLMs with tool use and memory. The ReAct pattern structures Thought → Action → Observation cycles. Tools include web search, code execution, file I/O, and API calls. Memory types: in-context (short), external vector (long), episodic, and semantic.", {"topic": "ai_agents"}),
    ("kb_07", "Multi-agent systems assign specialized roles to cooperating agents. CrewAI uses Researcher/Writer/Critic patterns. LangGraph models agent workflows as stateful directed graphs with nodes and edges. Parallelism and specialization outperform single monolithic agents on complex tasks.", {"topic": "multi_agent"}),
    ("kb_08", "MLOps practices: experiment tracking with MLflow/W&B, model versioning, Docker for reproducible environments, CI/CD with GitHub Actions, model monitoring for drift. Feature stores and model registries standardize the ML lifecycle in production.", {"topic": "mlops"}),
    ("kb_09", "LangGraph is a stateful agent framework. Nodes are Python functions that receive and return state dicts. Edges define control flow including conditional branches and loops. Supports human-in-the-loop, persistence, and streaming. Compiles to a runnable graph.", {"topic": "langgraph"}),
    ("kb_10", "Text embeddings map semantically similar text to nearby vectors. Sentence-Transformers produce dense embeddings. Contrastive learning trains embeddings via positive/negative pairs. Used for semantic search, clustering, and classification.", {"topic": "embeddings"}),
    ("kb_11", "Fine-tuning adapts a pretrained model to a specific task using labeled data. LoRA injects low-rank matrices for efficient fine-tuning. QLoRA combines quantization with LoRA. PEFT methods update only a small fraction of weights.", {"topic": "fine_tuning"}),
    ("kb_12", "Autonomous agents operate in loop: perceive → reason → act → observe → repeat. Self-critique and reflection improve plan quality. Tool selection driven by task decomposition. Human oversight added as interrupt nodes in the graph.", {"topic": "autonomous_systems"}),
    ("kb_13", "Computer vision: classification, detection, segmentation. CNNs extract spatial features. Transfer learning from ImageNet-pretrained models (ResNet, EfficientNet) accelerates development. Data augmentation prevents overfitting.", {"topic": "computer_vision"}),
    ("kb_14", "NLP tasks: classification, NER, summarization, translation, QA. BERT-family excels at understanding; GPT-family at generation. BPE/WordPiece tokenization converts text to subword tokens. Attention heads capture different linguistic relationships.", {"topic": "nlp"}),
    ("kb_15", "Python ML ecosystem: NumPy, Pandas, Scikit-learn, PyTorch, Hugging Face Transformers, LangChain/LangGraph, ChromaDB, Streamlit. Virtual environments and requirements.txt manage dependencies. Key workflow: explore → preprocess → train → evaluate → deploy.", {"topic": "python_ecosystem"}),
]


def _get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection("ai_knowledge", embedding_function=ef)


def seed_knowledge_base() -> int:
    """Seeds DB with AI/ML documents. Skips if already seeded. Returns doc count."""
    collection = _get_collection()
    if collection.count() > 0:
        return collection.count()
    ids = [d[0] for d in SEED_DOCS]
    texts = [d[1] for d in SEED_DOCS]
    metadatas = [d[2] for d in SEED_DOCS]
    collection.add(documents=texts, ids=ids, metadatas=metadatas)
    return len(SEED_DOCS)


def search(query: str, n_results: int = 3) -> list[str]:
    """Semantic search over knowledge base. Returns list of matching document strings."""
    collection = _get_collection()
    if collection.count() == 0:
        seed_knowledge_base()
    n = min(n_results, collection.count())
    results = collection.query(query_texts=[query], n_results=n)
    return results["documents"][0] if results["documents"] else []


def add_document(doc_id: str, text: str, metadata: dict = None):
    """Add a custom document to the knowledge base."""
    collection = _get_collection()
    collection.add(documents=[text], ids=[doc_id], metadatas=[metadata or {}])
