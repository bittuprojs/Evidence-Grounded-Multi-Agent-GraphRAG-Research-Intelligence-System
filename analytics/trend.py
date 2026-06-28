from collections import defaultdict
from typing import List, Dict, Any


def year_distribution(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = defaultdict(int)

    for doc in docs:
        year = doc.get("year")
        if year:
            counts[str(year)] += 1

    return dict(sorted(counts.items(), key=lambda x: x[0]))


def domain_distribution(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = defaultdict(int)

    for doc in docs:
        domain = doc.get("domain", "General")
        counts[domain] += 1

    return dict(sorted(counts.items(), key=lambda x: x[0]))


def top_keywords(docs: List[Dict[str, Any]], top_k: int = 20) -> List[tuple]:
    freq = defaultdict(int)

    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
        "were", "into", "using", "use", "based", "paper", "study", "data",
        "model", "models", "method", "methods", "results", "analysis"
    }

    for doc in docs:
        text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
        for word in text.split():
            w = word.strip(".,()[]{}:;!?\"'")
            if len(w) < 4 or w in stopwords:
                continue
            freq[w] += 1

    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]