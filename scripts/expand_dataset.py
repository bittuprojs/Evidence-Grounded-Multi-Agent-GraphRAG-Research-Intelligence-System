import sys
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.arxiv_loader import fetch_arxiv_papers
from ingestion.cleaner import deduplicate_documents, filter_short_docs

BASE_DIR = ROOT
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PAPERS_FILE = PROCESSED_DIR / "papers.json"
STATS_FILE = PROCESSED_DIR / "stats.json"
QUERY_LOG_FILE = PROCESSED_DIR / "query_log.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_TARGETS = {
    "Medical": 120,
    "Finance": 120,
    "General": 120,
}

PER_QUERY_RESULTS = 50
QUERY_DELAY = 0.5

QUERY_PLAN = {
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
    "General": [
        "machine learning",
        "deep learning",
        "data mining",
        "pattern recognition",
        "optimization algorithms",
        "statistical learning",
        "information retrieval",
        "natural language processing",
        "computer vision",
        "probabilistic models",
        "graph algorithms",
        "scientific computing",
        "numerical methods",
        "pattern classification",
        "dimensionality reduction",
        "feature engineering",
        "unsupervised learning",
        "supervised learning",
        "multi task learning",
        "representation learning",
        "time series analysis",
        "data analysis methods",
        "artificial intelligence",
        "computational methods",
        "applied machine learning",
        "algorithm design",
        "data science",
        "knowledge discovery",
        "statistics and learning",
        "computational statistics",
        "quantum computing applications",
        "computational physics methods",
        "materials science simulation",
        "particle physics data analysis",
        "condensed matter physics modeling",
        "plasma physics simulation",
        "astrophysics computational methods",
        "statistical mechanics algorithms",
        "computational biology methods",
        "bioinformatics analysis",
        "genomics data processing",
        "protein structure prediction",
        "systems biology modeling",
        "phylogenetic analysis algorithms",
        "metabolic network analysis",
        "single cell sequencing analysis",
        "chemical informatics",
        "drug discovery computational",
        "molecular modeling techniques",
        "chemical reaction prediction",
        "quantum chemistry calculations",
        "cheminformatics databases",
        "catalysis computational design",
        "spectroscopy data analysis",
         "climate modeling algorithms",
        "environmental data analysis",
        "pollution monitoring systems",
        "ecosystem modeling methods",
        "carbon cycle simulation",
        "biodiversity computational analysis",
        "water quality prediction models",
        "atmospheric chemistry modeling",
        "geological data processing",
        "seismic analysis algorithms",
        "mineral exploration methods",
        "hydrogeology modeling",
        "geospatial analysis techniques",
        "remote sensing applications",
        "geological simulation software",
        "natural disaster prediction",
        "brain imaging analysis",
        "neural network modeling",
        "computational neuroscience",
        "neuroinformatics methods",
        "cognitive modeling algorithms",
        "brain connectivity analysis",
        "neural signal processing",
        "behavioral data analysis",
         "numerical analysis methods",
        "optimization algorithms",
        "statistical modeling techniques",
        "machine learning mathematics",
        "graph theory applications",
        "linear algebra computational",
        "differential equations solving",
        "network analysis mathematics",
         "scientific data visualization",
        "research data management",
        "statistical analysis methods",
        "experimental design optimization",
        "scientific computing frameworks",
        "big data scientific applications",
        "research reproducibility methods",
        "scientific workflow automation"
    ],
}


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def make_doc_key(doc: Dict) -> str:
    title = doc.get("title", "").strip().lower()
    text = (doc.get("text", "") or doc.get("abstract", "")).strip().lower()[:200]
    base = f"{title}||{text}".encode("utf-8")
    return hashlib.md5(base).hexdigest()


def load_existing_papers() -> List[Dict]:
    if not PAPERS_FILE.exists():
        return []
    with open(PAPERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def count_domains(docs: List[Dict]) -> Dict[str, int]:
    counts = {}
    for doc in docs:
        domain = doc.get("domain", "General")
        counts[domain] = counts.get(domain, 0) + 1
    return counts


def safe_fetch(query: str, domain: str, max_results: int = PER_QUERY_RESULTS, retries: int = 3):
    last_error = None
    for attempt in range(retries + 1):
        try:
            docs = fetch_arxiv_papers(query=query, max_results=max_results, domain=domain)
            return docs
        except Exception as e:
            last_error = e
            print(f"ArXiv fetch error for '{query}' [{domain}] attempt {attempt + 1}: {e}")
            time.sleep(2 + attempt * 2)
    print(f"Skipping '{query}' after retries. Last error: {last_error}")
    return []


def expand_domain(existing_docs: List[Dict], domain: str, target_count: int, seen: set, query_log: List[Dict]) -> List[Dict]:
    current_count = sum(1 for d in existing_docs if d.get("domain") == domain)
    new_docs = []

    print("\n" + "=" * 60)
    print(f"EXPANDING DOMAIN: {domain}")
    print(f"Current count: {current_count}")
    print(f"Target count: {target_count}")
    print("=" * 60)

    if current_count >= target_count:
        print(f"{domain} already meets the target.")
        return new_docs

    for query in QUERY_PLAN[domain]:
        if current_count + len(new_docs) >= target_count:
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
            new_docs.append(doc)
            added += 1

        query_log.append({
            "domain": domain,
            "query": query,
            "returned": len(docs),
            "added": added
        })

        print(f"  returned={len(docs)} added={added} new_total_for_domain={current_count + len(new_docs)}")
        time.sleep(QUERY_DELAY)

    return new_docs


def main():
    existing_docs = load_existing_papers()
    existing_docs = deduplicate_documents(existing_docs)
    existing_docs = filter_short_docs(existing_docs, min_chars=200)

    seen = set(make_doc_key(doc) for doc in existing_docs)

    domain_counts = count_domains(existing_docs)
    query_log = []

    print("=" * 60)
    print("EXPANDING SAVED DATASET")
    print("=" * 60)
    print(f"Existing total papers: {len(existing_docs)}")
    print(f"Existing domain counts: {domain_counts}")

    all_new_docs = []

    for domain, target in DOMAIN_TARGETS.items():
        added_docs = expand_domain(existing_docs + all_new_docs, domain, target, seen, query_log)
        all_new_docs.extend(added_docs)

        # Save after each domain
        temp_docs = existing_docs + all_new_docs
        temp_docs = deduplicate_documents(temp_docs)
        temp_docs = filter_short_docs(temp_docs, min_chars=200)

        save_json(PAPERS_FILE, temp_docs)
        save_json(QUERY_LOG_FILE, query_log)

        temp_counts = count_domains(temp_docs)
        stats = {
            "total_unique_papers": len(temp_docs),
            "domain_counts": temp_counts,
            "domain_targets": DOMAIN_TARGETS
        }
        save_json(STATS_FILE, stats)

        print(f"\nSaved after domain {domain}")
        print(f"Total papers now: {len(temp_docs)}")
        print(f"Domain counts now: {temp_counts}")

    final_docs = existing_docs + all_new_docs
    final_docs = deduplicate_documents(final_docs)
    final_docs = filter_short_docs(final_docs, min_chars=200)

    final_counts = count_domains(final_docs)
    stats = {
        "total_unique_papers": len(final_docs),
        "domain_counts": final_counts,
        "domain_targets": DOMAIN_TARGETS
    }

    save_json(PAPERS_FILE, final_docs)
    save_json(STATS_FILE, stats)
    save_json(QUERY_LOG_FILE, query_log)

    print("\n" + "=" * 60)
    print("EXPANSION COMPLETE")
    print("=" * 60)
    print(f"Final total unique papers: {len(final_docs)}")
    print(f"Final domain counts: {final_counts}")
    print(f"Saved: {PAPERS_FILE}")
    print(f"Saved: {STATS_FILE}")
    print(f"Saved: {QUERY_LOG_FILE}")


if __name__ == "__main__":
    main()