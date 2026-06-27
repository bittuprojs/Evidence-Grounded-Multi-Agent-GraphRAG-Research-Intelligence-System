import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.cleaner import deduplicate_documents, filter_short_docs, make_doc_key

BASE_DIR = ROOT
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FOUNDATION_FILE = PROCESSED_DIR / "papers_with_foundations.json"
SOURCE_FILE = PROCESSED_DIR / "papers.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FOUNDATION_DOCS = [
    {
        "doc_id": "foundation_ml_basics",
        "title": "Machine Learning Basics: A Foundational Overview",
        "text": """
Machine learning is a field of artificial intelligence in which computers learn patterns from data instead of being explicitly programmed for every rule.
The goal is to build models that can make predictions, classify inputs, detect structure, or support decision-making.

The main categories are supervised learning, unsupervised learning, and reinforcement learning.
In supervised learning, the model learns from labeled examples.
In unsupervised learning, the system tries to discover patterns without labels.
In reinforcement learning, an agent learns by interacting with an environment and receiving rewards.

Common machine learning workflows include data collection, cleaning, feature preparation, model training, validation, testing, and deployment.
Important ideas include generalization, overfitting, underfitting, bias-variance tradeoff, and evaluation metrics.

Machine learning is used in recommendation systems, fraud detection, healthcare decision support, search ranking, image analysis, and language processing.
A strong machine learning system is not just accurate; it must also be robust, explainable, and reliable.
        """.strip(),
        "authors": ["OpenAI"],
        "year": 2024,
        "source": "foundation_pack",
        "domain": "General",
        "url": None,
        "metadata": {"type": "foundation", "topic": "ml_basics"},
    },
    {
        "doc_id": "foundation_deep_learning",
        "title": "Deep Learning Basics and Neural Networks",
        "text": """
Deep learning is a subfield of machine learning based on neural networks with multiple layers.
These layered models learn hierarchical representations from data.
Early layers often learn simple patterns, while deeper layers learn more abstract structures.

A neural network is trained by comparing predictions with true targets, computing a loss, and updating weights using backpropagation and gradient-based optimization.
Important concepts include activations, loss functions, optimization, regularization, dropout, batch normalization, and learning rate scheduling.

Deep learning is especially effective for images, speech, natural language, and large-scale sequence modeling.
Its success comes from representation learning, where the model automatically learns useful features from raw data.

However, deep learning can require large datasets, high compute, careful tuning, and strong evaluation to avoid overfitting or unstable behavior.
        """.strip(),
        "authors": ["OpenAI"],
        "year": 2024,
        "source": "foundation_pack",
        "domain": "General",
        "url": None,
        "metadata": {"type": "foundation", "topic": "deep_learning"},
    },
    {
        "doc_id": "foundation_transformers",
        "title": "Transformers, Attention, and Large Language Models",
        "text": """
Transformers are neural architectures that rely on attention mechanisms rather than recurrence.
Attention allows a model to weigh which input tokens are most relevant when producing a representation or prediction.

The transformer architecture uses self-attention, multi-head attention, positional information, and feed-forward layers.
It is highly parallelizable and works well on long sequences.

Transformers became central to modern natural language processing and later to vision, audio, code, and multimodal systems.
Large language models are usually built on transformer backbones and trained on large corpora to predict tokens and learn general language behavior.

Important practical ideas include context window, prompt design, instruction tuning, retrieval augmentation, and grounding with evidence.
        """.strip(),
        "authors": ["OpenAI"],
        "year": 2024,
        "source": "foundation_pack",
        "domain": "AI/ML",
        "url": None,
        "metadata": {"type": "foundation", "topic": "transformers"},
    },
    {
        "doc_id": "foundation_rag",
        "title": "Retrieval-Augmented Generation (RAG) Basics",
        "text": """
Retrieval-augmented generation combines search and generation.
Instead of relying only on the language model memory, the system first retrieves relevant documents and then generates an answer using that evidence.

A RAG pipeline usually contains:
1) query understanding,
2) retrieval,
3) reranking,
4) context assembly,
5) answer generation,
6) citation or source reporting.

RAG improves factual grounding, reduces hallucinations, and makes answers easier to verify.
It is especially useful for research assistants, enterprise search, technical support, and knowledge-heavy workflows.

The quality of a RAG system depends on chunking, embedding quality, retrieval strategy, reranking, prompt design, and evidence selection.
A poor retrieval step usually produces a poor answer even if the generator is strong.
        """.strip(),
        "authors": ["OpenAI"],
        "year": 2024,
        "source": "foundation_pack",
        "domain": "General",
        "url": None,
        "metadata": {"type": "foundation", "topic": "rag"},
    },
    {
        "doc_id": "foundation_embeddings",
        "title": "Embeddings and Semantic Search",
        "text": """
Embeddings are vector representations of text that place similar meanings near each other in vector space.
They are useful because they let systems perform semantic search, clustering, retrieval, and similarity comparison.

Semantic search works by embedding the query and the documents, then ranking documents by vector similarity.
This is more flexible than keyword matching because it can capture meaning rather than exact word overlap.

Embeddings are critical for modern retrieval systems, document search, recommendation, and RAG.
Good retrieval quality depends on clean text, meaningful chunks, and an embedding model that matches the task domain.
        """.strip(),
        "authors": ["OpenAI"],
        "year": 2024,
        "source": "foundation_pack",
        "domain": "General",
        "url": None,
        "metadata": {"type": "foundation", "topic": "embeddings"},
    },
    {
        "doc_id": "foundation_evaluation",
        "title": "Evaluation, Grounding, and Trustworthy Answers",
        "text": """
A trustworthy research assistant must evaluate whether an answer is grounded in evidence.
Useful signals include source coverage, citation quality, retrieval confidence, contradiction checks, and abstention when evidence is weak.

Grounding means that the answer can be traced back to retrieved documents.
If the evidence is insufficient, the assistant should say so rather than inventing information.
This is important for academic use, research workflows, and technical decision support.

Common evaluation dimensions include accuracy, relevance, faithfulness, completeness, latency, and citation coverage.
A strong system balances response quality with transparency and reliability.
        """.strip(),
        "authors": ["OpenAI"],
        "year": 2024,
        "source": "foundation_pack",
        "domain": "General",
        "url": None,
        "metadata": {"type": "foundation", "topic": "evaluation"},
    },
    {
        "doc_id": "foundation_research_workflow",
        "title": "Research Comparison and Literature Review Workflow",
        "text": """
A research assistant should support comparison and literature review tasks.
Comparison means identifying similarities and differences between two methods, models, or papers.
Literature review means summarizing several sources, grouping them by theme, and highlighting gaps.

A useful workflow is:
- understand the question,
- retrieve evidence,
- group by subtopic,
- compare methods and results,
- detect limitations,
- summarize research gaps.

This turns a simple search tool into a real research intelligence assistant.
        """.strip(),
        "authors": ["OpenAI"],
        "year": 2024,
        "source": "foundation_pack",
        "domain": "General",
        "url": None,
        "metadata": {"type": "foundation", "topic": "workflow"},
    },
]

def load_existing_docs():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Missing source dataset: {SOURCE_FILE}")
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

def build_foundation_pack():
    docs = load_existing_docs()

    existing_keys = {make_doc_key(d.get("title", ""), d.get("text", "") or d.get("abstract", "")) for d in docs}
    new_docs = []

    for doc in FOUNDATION_DOCS:
        key = make_doc_key(doc.get("title", ""), doc.get("text", ""))
        if key not in existing_keys:
            new_docs.append(doc)
            existing_keys.add(key)

    merged = docs + new_docs
    merged = deduplicate_documents(merged)
    merged = filter_short_docs(merged, min_chars=120)

    save_json(FOUNDATION_FILE, merged)

    print("=" * 60)
    print("FOUNDATION PACK BUILT")
    print("=" * 60)
    print(f"Original docs: {len(docs)}")
    print(f"Added foundation docs: {len(new_docs)}")
    print(f"Total docs: {len(merged)}")
    print(f"Saved: {FOUNDATION_FILE}")

if __name__ == "__main__":
    build_foundation_pack()