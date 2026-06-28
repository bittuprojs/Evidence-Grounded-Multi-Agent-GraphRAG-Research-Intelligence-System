from collections import defaultdict
from typing import List, Dict, Any


class ResearchGapDetector:
    def detect_gaps(self, docs: List[Dict[str, Any]], topic: str = "") -> Dict[str, Any]:
        """
        Very simple heuristic gap detector.
        Looks for missing or weakly covered themes.
        """
        if not docs:
            return {
                "topic": topic,
                "gaps": ["No documents available for analysis."],
                "covered_keywords": [],
                "domain_focus": {}
            }

        keyword_freq = defaultdict(int)
        domain_freq = defaultdict(int)

        important_terms = [
            "survey", "review", "benchmark", "evaluation", "dataset",
            "robust", "explainable", "interpretability", "factual",
            "hallucination", "uncertainty", "bias", "fairness",
            "generalization", "scalability", "efficiency", "latency",
            "retrieval", "reranking", "citation", "grounding"
        ]

        for doc in docs:
            domain = doc.get("domain", "General")
            domain_freq[domain] += 1

            text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
            for term in important_terms:
                if term in text:
                    keyword_freq[term] += 1

        gaps = []

        if keyword_freq.get("hallucination", 0) < 5:
            gaps.append("Limited explicit focus on hallucination detection and mitigation.")

        if keyword_freq.get("citation", 0) < 5 or keyword_freq.get("grounding", 0) < 5:
            gaps.append("Citation grounding and source-traceability are underrepresented.")

        if keyword_freq.get("benchmark", 0) < 5 or keyword_freq.get("evaluation", 0) < 5:
            gaps.append("Benchmarking and evaluation studies are relatively sparse.")

        if keyword_freq.get("uncertainty", 0) < 5:
            gaps.append("Uncertainty estimation and confidence calibration are not well covered.")

        if keyword_freq.get("fairness", 0) < 5 or keyword_freq.get("bias", 0) < 5:
            gaps.append("Fairness and bias analysis appear underrepresented.")

        if keyword_freq.get("interpretability", 0) < 5 or keyword_freq.get("explainable", 0) < 5:
            gaps.append("Interpretability and explainability are not deeply covered.")

        if not gaps:
            gaps.append("No strong gap detected from the current corpus slice.")

        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:15]

        return {
            "topic": topic,
            "gaps": gaps,
            "covered_keywords": top_keywords,
            "domain_focus": dict(domain_freq),
        }