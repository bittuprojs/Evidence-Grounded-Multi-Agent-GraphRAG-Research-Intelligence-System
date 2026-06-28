from typing import List, Dict, Optional
import re
import statistics


class EvaluationSuite:
    """
    Runs benchmark-style checks over the answer engine.
    Metrics:
    - support_score
    - contradiction_score
    - retrieval_precision
    - confidence_score
    - overall_quality
    """

    DEFAULT_CASES = [
        {
            "name": "Machine Learning Basics",
            "question": "What is machine learning?",
            "domain": "All Domains",
            "expected_keywords": [
                "machine learning",
                "supervised learning",
                "unsupervised learning",
                "deep learning",
            ],
        },
        {
            "name": "Retrieval Augmented Generation",
            "question": "What is retrieval augmented generation?",
            "domain": "All Domains",
            "expected_keywords": [
                "retrieval augmented generation",
                "retrieval",
                "generation",
                "grounding",
            ],
        },
        {
            "name": "Transformer Models",
            "question": "What are transformers in natural language processing?",
            "domain": "All Domains",
            "expected_keywords": [
                "transformer",
                "attention",
                "natural language processing",
            ],
        },
        {
            "name": "BERT vs RoBERTa",
            "question": "Compare BERT and RoBERTa",
            "domain": "All Domains",
            "expected_keywords": [
                "bert",
                "roberta",
                "transformer",
            ],
        },
        {
            "name": "RAG Limitations",
            "question": "What are the limitations of retrieval augmented generation?",
            "domain": "All Domains",
            "expected_keywords": [
                "hallucination",
                "retrieval",
                "grounding",
                "context",
            ],
        },
    ]

    def __init__(self, answer_engine):
        self.answer_engine = answer_engine

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower()).strip()

    def _retrieval_precision(self, sources: List[Dict], expected_keywords: List[str]) -> float:
        if not sources:
            return 0.0

        if not expected_keywords:
            return 0.5

        haystack = []
        for s in sources:
            haystack.append(self._normalize_text(s.get("title", "")))
            haystack.append(self._normalize_text(s.get("excerpt", "")))

        combined = " ".join(haystack)

        hits = 0
        for kw in expected_keywords:
            if self._normalize_text(kw) in combined:
                hits += 1

        return round(hits / max(len(expected_keywords), 1), 4)

    def _answer_relevance(self, answer: str, question: str) -> float:
        """
        Lightweight overlap heuristic between answer and question.
        """
        answer_tokens = set(re.findall(r"[a-z0-9]+", self._normalize_text(answer)))
        question_tokens = set(re.findall(r"[a-z0-9]+", self._normalize_text(question)))

        if not question_tokens:
            return 0.0

        overlap = len(answer_tokens & question_tokens)
        return round(overlap / max(len(question_tokens), 1), 4)

    def run_case(
        self,
        question: str,
        domain: Optional[str] = None,
        expected_keywords: Optional[List[str]] = None
    ) -> Dict:
        result = self.answer_engine.answer(question, domain=domain)

        sources = result.get("sources", [])
        verification = result.get("verification", {})
        evidence = verification.get("evidence", {}) if isinstance(verification, dict) else {}
        contradiction = verification.get("contradiction", {}) if isinstance(verification, dict) else {}

        support_score = float(evidence.get("support_score", 0.0) or 0.0)
        contradiction_score = float(contradiction.get("contradiction_score", 0.0) or 0.0)
        retrieval_precision = self._retrieval_precision(sources, expected_keywords or [])
        answer_relevance = self._answer_relevance(result.get("answer", ""), question)
        confidence_score = float(result.get("confidence_score", 0.0) or 0.0)

        overall_quality = (
            0.35 * support_score
            + 0.25 * (1.0 - contradiction_score)
            + 0.20 * retrieval_precision
            + 0.10 * answer_relevance
            + 0.10 * confidence_score
        )

        overall_quality = round(max(0.0, min(overall_quality, 1.0)), 4)

        if overall_quality >= 0.75:
            status = "excellent"
        elif overall_quality >= 0.55:
            status = "good"
        elif overall_quality >= 0.35:
            status = "fair"
        else:
            status = "weak"

        return {
            "question": question,
            "domain": domain or "All Domains",
            "intent": result.get("intent", ""),
            "confidence": result.get("confidence", ""),
            "confidence_score": confidence_score,
            "support_score": round(support_score, 4),
            "contradiction_score": round(contradiction_score, 4),
            "retrieval_precision": retrieval_precision,
            "answer_relevance": answer_relevance,
            "overall_quality": overall_quality,
            "status": status,
            "citation_count": len(sources),
            "sources": sources,
            "answer": result.get("answer", ""),
            "verification": verification,
            "processing_time": result.get("processing_time", 0.0),
        }

    def run_benchmark(self, cases: Optional[List[Dict]] = None) -> Dict:
        cases = cases or self.DEFAULT_CASES

        rows = []
        for case in cases:
            row = self.run_case(
                question=case["question"],
                domain=case.get("domain"),
                expected_keywords=case.get("expected_keywords", []),
            )
            row["case_name"] = case.get("name", case["question"])
            rows.append(row)

        summary = self._summarize(rows)
        return {
            "summary": summary,
            "results": rows,
        }

    def _summarize(self, rows: List[Dict]) -> Dict:
        if not rows:
            return {
                "cases": 0,
                "avg_support_score": 0.0,
                "avg_contradiction_score": 0.0,
                "avg_retrieval_precision": 0.0,
                "avg_answer_relevance": 0.0,
                "avg_confidence_score": 0.0,
                "avg_overall_quality": 0.0,
                "pass_rate": 0.0,
            }

        avg_support = statistics.mean(r["support_score"] for r in rows)
        avg_contradiction = statistics.mean(r["contradiction_score"] for r in rows)
        avg_precision = statistics.mean(r["retrieval_precision"] for r in rows)
        avg_relevance = statistics.mean(r["answer_relevance"] for r in rows)
        avg_confidence = statistics.mean(r["confidence_score"] for r in rows)
        avg_overall = statistics.mean(r["overall_quality"] for r in rows)

        passed = sum(1 for r in rows if r["status"] in {"excellent", "good"})
        pass_rate = passed / len(rows)

        return {
            "cases": len(rows),
            "avg_support_score": round(avg_support, 4),
            "avg_contradiction_score": round(avg_contradiction, 4),
            "avg_retrieval_precision": round(avg_precision, 4),
            "avg_answer_relevance": round(avg_relevance, 4),
            "avg_confidence_score": round(avg_confidence, 4),
            "avg_overall_quality": round(avg_overall, 4),
            "pass_rate": round(pass_rate, 4),
        }