import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict

# Make local imports work both with:
# python -m scripts.build_dataset
# and
# python scripts/build_dataset.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.arxiv_loader import fetch_arxiv_papers
from ingestion.pdf_loader import load_pdfs_from_folder
from ingestion.cleaner import deduplicate_documents, filter_short_docs

BASE_DIR = ROOT
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PAPERS = 450
PER_QUERY_RESULTS = 50
QUERY_DELAY = 0.4

QUERY_PLAN = {
    "AI/ML": [
        "machine learning fundamentals",
        "deep learning",
        "neural networks",
        "transformers in machine learning",
        "retrieval augmented generation",
        "RAG evaluation",
        "hallucination detection in LLMs",
        "grounded question answering",
        "dense retrieval NLP",
        "document reranking transformers",
        "large language models",
        "prompt engineering",
        "instruction tuning",
        "multi-agent systems",
        "graph neural networks",
        "vision transformers",
        "reinforcement learning",
        "generative AI",
        "embeddings semantic search",
        "information retrieval",
        "llm reasoning",
        "self-rag",
        "hybrid retrieval",
        "semantic search",
        "question answering systems",
        "text summarization",
        "model alignment",
        "factual consistency",
        "prompt optimization",
        "retrieval evaluation",
        "support vector machines",
        "decision trees",
        "bayesian learning",
        "computer vision",
        "natural language processing",
        "speech recognition",
        "knowledge graphs",
        "recommendation systems",
        "few-shot learning",
        "self-supervised learning",
        "multimodal learning",
        "continual learning",
        "federated learning",
        "adversarial machine learning",
        "explainable AI",
        "neural architecture search",
        "optimization in deep learning",
        "language model evaluation",
        "retrieval systems",
        "decision support systems",
    ],
    "Medical": [
        "medical image analysis",
        "clinical decision support",
        "healthcare artificial intelligence",
        "diagnostic imaging AI",
        "telemedicine applications",
        "electronic health records",
        "patient monitoring systems",
        "healthcare data analytics",
        "medical segmentation deep learning",
        "radiology AI",
        "pathology AI",
        "biomedical text mining",
        "drug discovery machine learning",
        "clinical trial prediction",
        "medical diagnosis accuracy",
        "health informatics",
        "computer aided diagnosis",
        "medical report generation",
        "biomedical image segmentation",
        "clinical NLP",
        "disease prediction model",
        "medical question answering",
        "brain tumor segmentation",
        "covid diagnosis AI",
        "medical recommendation systems",
        "AI in pathology",
        "MRI segmentation",
        "clinical language models",
        "biomedical question answering",
        "genomic prediction",
        "disease classification",
    ],
    "Finance": [
        "financial machine learning",
        "algorithmic trading systems",
        "financial risk management",
        "blockchain financial applications",
        "cryptocurrency market analysis",
        "market prediction models",
        "economic forecasting methods",
        "financial data analysis",
        "investment strategy optimization",
        "portfolio optimization",
        "derivatives pricing models",
        "high frequency trading",
        "fraud detection machine learning",
        "credit scoring models",
        "volatility forecasting",
        "fintech machine learning",
        "stock price prediction",
        "loan default prediction",
        "sentiment analysis finance",
        "financial time series forecasting",
        "risk modeling",
        "quantitative finance",
        "financial forecasting",
        "banking analytics",
        "insurance risk modeling",
        "time series forecasting finance",
        "AI in economics",
        "economic prediction models",
    ],
    "Science": [
        "scientific machine learning",
        "computational physics methods",
        "materials science simulation",
        "particle physics data analysis",
        "astrophysics computational methods",
        "computational biology methods",
        "bioinformatics analysis",
        "genomics data processing",
        "protein structure prediction",
        "chemical informatics",
        "drug discovery computational",
        "climate modeling algorithms",
        "environmental data analysis",
        "geospatial analysis techniques",
        "remote sensing applications",
        "computational neuroscience",
        "numerical analysis methods",
        "optimization algorithms",
        "scientific computing",
        "data-driven science",
        "physics informed machine learning",
        "molecular modeling",
        "quantum computing",
        "robotics",
        "computational chemistry",
        "scientific visualization",
        "climate science AI",
        "space science data analysis",
        "renewable energy optimization",
        "smart grid systems",
        "scientific simulations",
    ],
}


def save_json(path: Path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )


def make_doc_key(doc: Dict) -> str:
    """
    Stable duplicate key for cross-domain deduplication.
    """
    title = doc.get("title", "").strip().lower()
    text = (doc.get("text", "") or doc.get("abstract", "")).strip().lower()[:200]
    base = f"{title}||{text}".encode("utf-8")
    return hashlib.md5(base).hexdigest()


def safe_fetch(query: str, domain: str, max_results: int = PER_QUERY_RESULTS, retries: int = 3):
    """
    Fetch from ArXiv with retry handling for 429/503/rate-limit issues.
    """
    last_error = None

    for attempt in range(retries + 1):
        try:
            docs = fetch_arxiv_papers(query=query, max_results=max_results, domain=domain)
            return docs
        except Exception as e:
            last_error = e
            print(f"ArXiv fetch error for query '{query}': {e}")
            wait_time = 1.5 + attempt * 2.0
            time.sleep(wait_time)

    print(f"Giving up on '{query}' after retries. Last error: {last_error}")
    return []


def load_local_pdfs() -> List[Dict]:
    pdf_folder = RAW_DIR / "pdfs"
    if not pdf_folder.exists():
        return []
    return load_pdfs_from_folder(str(pdf_folder), domain="General")


def build_dataset() -> List[Dict]:
    all_docs: List[Dict] = []
    seen = set()
    query_log: List[Dict] = []

    print("=" * 60)
    print("BUILDING MULTI-DOMAIN RESEARCH DATASET")
    print("=" * 60)
    print(f"Target unique papers: {TARGET_PAPERS}")
    print()

    # 1) Collect from ArXiv
    for domain, queries in QUERY_PLAN.items():
        print(f"\n=== DOMAIN: {domain} ===")

        for query in queries:
            if len(all_docs) >= TARGET_PAPERS:
                break

            print(f"Fetching: {query}")
            docs = safe_fetch(query=query, domain=domain, max_results=PER_QUERY_RESULTS)

            added = 0
            for doc in docs:
                doc["domain"] = domain
                doc["subdomain"] = query
                doc["text"] = doc.get("text", "") or doc.get("abstract", "")

                key = make_doc_key(doc)
                if key in seen:
                    continue

                seen.add(key)
                all_docs.append(doc)
                added += 1

            query_log.append({
                "domain": domain,
                "query": query,
                "returned": len(docs),
                "added": added
            })

            print(f"  returned={len(docs)} added={added} total={len(all_docs)}")
            time.sleep(QUERY_DELAY)

        if len(all_docs) >= TARGET_PAPERS:
            break

    # 2) Load local PDFs if present
    print("\nLoading local PDFs if any...")
    pdf_docs = load_local_pdfs()
    pdf_added = 0

    for doc in pdf_docs:
        key = make_doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        all_docs.append(doc)
        pdf_added += 1

    if pdf_added:
        print(f"Added {pdf_added} local PDF documents")

    # 3) Clean and filter
    all_docs = deduplicate_documents(all_docs)
    all_docs = filter_short_docs(all_docs, min_chars=200)

    # Optional cap if corpus gets too large
    if len(all_docs) > 500:
        all_docs = all_docs[:500]

    # 4) Save outputs
    papers_file = PROCESSED_DIR / "papers.json"
    stats_file = PROCESSED_DIR / "stats.json"
    query_log_file = PROCESSED_DIR / "query_log.json"

    save_json(papers_file, all_docs)
    save_json(query_log_file, query_log)

    # 5) Stats
    domain_counts = {}
    year_counts = {}
    category_counts = {}

    for doc in all_docs:
        domain = doc.get("domain", "General")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        year = doc.get("year")
        if year:
            year_counts[str(year)] = year_counts.get(str(year), 0) + 1

        for cat in doc.get("metadata", {}).get("categories", []):
            category_counts[cat] = category_counts.get(cat, 0) + 1

    stats = {
        "target_papers": TARGET_PAPERS,
        "total_unique_papers": len(all_docs),
        "query_count": len(query_log),
        "domain_counts": domain_counts,
        "year_counts": year_counts,
        "top_arxiv_categories": dict(
            sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        ),
    }
    save_json(stats_file, stats)

    print("\n" + "=" * 60)
    print("DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"Total unique papers: {len(all_docs)}")
    print(f"Saved: {papers_file}")
    print(f"Saved: {stats_file}")
    print(f"Saved: {query_log_file}")

    print("\nDomain counts:")
    for d, c in sorted(domain_counts.items()):
        print(f"  {d}: {c}")

    if len(all_docs) < TARGET_PAPERS:
        print("\nWARNING: Target not fully reached.")
        print("Add more queries to QUERY_PLAN and rerun.")
    else:
        print("\nTarget reached successfully.")

    return all_docs


if __name__ == "__main__":
    build_dataset()