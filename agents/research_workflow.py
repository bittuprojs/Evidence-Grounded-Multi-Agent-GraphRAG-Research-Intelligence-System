import time
import re
from typing import Dict, List, Optional

from retrieval.query_router import QueryRouter
from generation.citation_manager import CitationManager
from generation.evidence_checker import EvidenceChecker
from generation.contradiction_checker import ContradictionChecker
from analytics.gap_detector import ResearchGapDetector


class ResearchWorkflow:
    """
    Multi-agent style orchestration:
    1) Query planning
    2) Retrieval
    3) Verification
    4) Synthesis
    5) Citations
    6) Analytics
    """

    def __init__(self, retriever, llm_fn=None, all_docs=None):
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.all_docs = all_docs or []

        self.router = QueryRouter()
        self.citations = CitationManager()
        self.evidence_checker = EvidenceChecker()
        self.contradiction_checker = ContradictionChecker()
        self.gap_detector = ResearchGapDetector()

    def _normalize_score(self, score):
        try:
            score = float(score)
        except Exception:
            return 0.0
        if score < 0:
            score = 1.0 / (1.0 + abs(score))
        return max(0.0, min(score, 1.0))

    def _rank(self, query: str, chunks: List[Dict], intent: str) -> List[Dict]:
        if not chunks:
            return []

        broad = intent == "broad_qa"
        ranked = []

        broad_terms = [
            "survey", "review", "overview", "introduction",
            "foundations", "fundamentals", "tutorial", "guide", "baseline"
        ]

        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))

        for c in chunks:
            base_score = self._normalize_score(c.get("score", 0.0))
            title = (c.get("title", "") or "").lower()
            text = (c.get("text", "") or "").lower()
            domain = (c.get("domain", "") or "").lower()

            bonus = 0.0

            if broad:
                if any(term in title for term in broad_terms):
                    bonus += 0.35
                if any(term in text[:600] for term in broad_terms):
                    bonus += 0.15
                if "machine learning" in title or "deep learning" in title or "artificial intelligence" in title:
                    bonus += 0.10
                year = c.get("year")
                try:
                    if year and int(year) <= 2022:
                        bonus += 0.05
                except Exception:
                    pass
                if domain in {"ai/ml", "general"}:
                    bonus += 0.05
            else:
                t_tokens = set(re.findall(r"[a-z0-9]+", title))
                overlap = len(q_tokens & t_tokens)
                bonus += min(0.20, overlap * 0.03)

            item = dict(c)
            item["score"] = min(1.0, base_score + bonus)
            ranked.append(item)

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def _build_prompt(self, question: str, chunks: List[Dict]) -> str:
        evidence = []
        for i, c in enumerate(chunks[:5], 1):
            evidence.append(
                f"[Evidence {i}]\n"
                f"Title: {c.get('title', 'Unknown')}\n"
                f"Source: {c.get('source', 'unknown')}\n"
                f"Domain: {c.get('domain', 'General')}\n"
                f"Text: {c.get('text', '')[:1200]}\n"
            )

        return (
            "You are a research assistant.\n"
            "Answer only from the evidence.\n"
            "If evidence is weak, say so clearly.\n"
            "Give a short, readable research answer.\n\n"
            f"Question: {question}\n\n"
            + "\n".join(evidence)
        )

    def _fallback_answer(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "I could not find enough evidence to answer reliably."

        lines = []
        for c in chunks[:3]:
            title = c.get("title", "Unknown title")
            excerpt = c.get("text", "")[:240].strip()
            lines.append(f"- {title}: {excerpt}...")

        return "Based on the best available evidence:\n" + "\n".join(lines)

    def _retrieve(self, plan) -> Dict:
        """
        Returns a retrieval bundle:
        - mode
        - chunks
        - optional compare side bundles
        """
        if plan.intent == "compare" and len(plan.entities) >= 2:
            left_name, right_name = plan.entities[0], plan.entities[1]

            left_chunks = self.retriever.search(
                left_name,
                top_k=7,
                domain=plan.domain,
                must_contain_terms=[left_name]
            )
            right_chunks = self.retriever.search(
                right_name,
                top_k=7,
                domain=plan.domain,
                must_contain_terms=[right_name]
            )

            if not left_chunks:
                fallback_left = self.retriever.search(left_name, top_k=7, domain=plan.domain)
                left_chunks = [
                    c for c in fallback_left
                    if left_name.lower() in f"{c.get('title', '')} {c.get('text', '')}".lower()
                ]

            if not right_chunks:
                fallback_right = self.retriever.search(right_name, top_k=7, domain=plan.domain)
                right_chunks = [
                    c for c in fallback_right
                    if right_name.lower() in f"{c.get('title', '')} {c.get('text', '')}".lower()
                ]

            return {
                "mode": "compare",
                "left_name": left_name,
                "right_name": right_name,
                "left_chunks": left_chunks,
                "right_chunks": right_chunks,
                "chunks": left_chunks + right_chunks
            }

        if plan.intent == "broad_qa":
            chunks = self.retriever.search(
                plan.rewritten_query,
                top_k=plan.top_k,
                domain=plan.domain,
                source="foundation_pack"
            )
            if len(chunks) < 3:
                extra = self.retriever.search(
                    plan.rewritten_query,
                    top_k=plan.top_k,
                    domain=plan.domain,
                    source=None
                )
                seen = {c["chunk_id"] for c in chunks}
                for c in extra:
                    if c["chunk_id"] not in seen:
                        chunks.append(c)

            return {
                "mode": "broad_qa",
                "chunks": chunks
            }

        chunks = self.retriever.search(
            plan.rewritten_query,
            top_k=plan.top_k,
            domain=plan.domain
        )

        return {
            "mode": "specific_qa",
            "chunks": chunks
        }

    def _synthesize_normal(self, question: str, plan, chunks: List[Dict]) -> str:
        prompt = self._build_prompt(question, chunks)

        if self.llm_fn:
            try:
                return self.llm_fn(prompt)
            except Exception:
                return self._fallback_answer(chunks)

        return self._fallback_answer(chunks)

    def _synthesize_compare(self, left_name: str, right_name: str, left_chunks: List[Dict], right_chunks: List[Dict]) -> str:
        left_block = "\n".join(
            [f"- {c.get('title', 'Unknown')}: {c.get('text', '')[:180]}..." for c in left_chunks[:3]]
        )
        right_block = "\n".join(
            [f"- {c.get('title', 'Unknown')}: {c.get('text', '')[:180]}..." for c in right_chunks[:3]]
        )

        prompt = f"""
You are comparing two research entities.

Left entity: {left_name}
Right entity: {right_name}

Use the evidence to produce:
1. Similarities
2. Differences
3. What each side contributes
4. A short final verdict

Evidence for {left_name}:
{left_block}

Evidence for {right_name}:
{right_block}
"""

        if self.llm_fn:
            try:
                return self.llm_fn(prompt)
            except Exception:
                pass

        if not left_chunks or not right_chunks:
            return (
                f"I could not find enough direct evidence in the current corpus to compare "
                f"{left_name} and {right_name} reliably."
            )

        return (
            f"Comparison of {left_name} and {right_name}:\n\n"
            f"{left_name}:\n{left_block}\n\n"
            f"{right_name}:\n{right_block}\n"
        )

    def run(self, question: str, domain: Optional[str] = None) -> Dict:
        start = time.time()

        plan = self.router.build_plan(question, domain=domain)
        retrieval = self._retrieve(plan)

        if retrieval["mode"] == "compare":
            left_chunks = retrieval.get("left_chunks", [])
            right_chunks = retrieval.get("right_chunks", [])
            answer_text = self._synthesize_compare(
                retrieval.get("left_name", ""),
                retrieval.get("right_name", ""),
                left_chunks,
                right_chunks
            )
            sources = []
            for c in left_chunks[:2] + right_chunks[:2]:
                sources.append({
                    "title": c.get("title"),
                    "year": c.get("year"),
                    "source": c.get("source"),
                    "url": c.get("url"),
                    "score": round(float(c.get("score", 0.0)), 4),
                    "excerpt": c.get("text", "")[:300],
                })
        else:
            chunks = self._rank(question, retrieval.get("chunks", []), plan.intent)
            answer_text = self._synthesize_normal(question, plan, chunks)
            sources = []
            for c in chunks[:5]:
                sources.append({
                    "title": c.get("title"),
                    "year": c.get("year"),
                    "source": c.get("source"),
                    "url": c.get("url"),
                    "score": round(float(c.get("score", 0.0)), 4),
                    "excerpt": c.get("text", "")[:300],
                })

        evidence_verification = self.evidence_checker.safety_label(
            answer_text,
            sources,
            question=question,
            intent=plan.intent
        )
        contradiction_verification = self.contradiction_checker.safety_label(
            answer_text,
            sources,
            question=question,
            intent=plan.intent
        )

        confidence = "medium"
        confidence_score = 0.5

        if evidence_verification["label"] == "supported" and contradiction_verification["label"] == "consistent":
            confidence = "high"
            confidence_score = 0.82
        elif evidence_verification["label"] == "caution" or contradiction_verification["label"] == "uncertain":
            confidence = "medium"
            confidence_score = 0.48
        else:
            confidence = "low"
            confidence_score = 0.20

        if plan.intent == "compare" and len(retrieval.get("left_chunks", [])) == 0 and len(retrieval.get("right_chunks", [])) == 0:
            confidence = "low"
            confidence_score = 0.10

        if not sources:
            citations = "No sources available."
        else:
            citations = self.citations.export_bibliography(sources)

        research_gaps = self.gap_detector.detect_gaps(self.all_docs, topic=question)

        source_domains = {}
        for s in sources:
            d = s.get("source", "unknown")
            source_domains[d] = source_domains.get(d, 0) + 1

        return {
            "question": question,
            "intent": plan.intent,
            "rewritten_query": plan.rewritten_query,
            "plan": {
                "intent": plan.intent,
                "domain": plan.domain,
                "rewritten_query": plan.rewritten_query,
                "top_k": plan.top_k,
                "entities": getattr(plan, "entities", [])
            },
            "retrieval": {
                "mode": retrieval.get("mode"),
                "left_name": retrieval.get("left_name"),
                "right_name": retrieval.get("right_name"),
                "source_domains": source_domains,
                "chunk_count": len(retrieval.get("chunks", [])),
            },
            "answer": answer_text,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "sources": sources,
            "citations": citations,
            "verification": {
                "evidence": evidence_verification,
                "contradiction": contradiction_verification,
            },
            "research_gaps": research_gaps,
            "processing_time": round(time.time() - start, 4),
        }