import time
import re

from retrieval.query_router import QueryRouter
from generation.citation_manager import CitationManager
from generation.evidence_checker import EvidenceChecker
from generation.contradiction_checker import ContradictionChecker
from analytics.gap_detector import ResearchGapDetector


class AnswerEngine:
    def __init__(self, retriever, llm_fn=None, all_docs=None):
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.router = QueryRouter()
        self.citations = CitationManager()
        self.evidence_checker = EvidenceChecker()
        self.contradiction_checker = ContradictionChecker()
        self.gap_detector = ResearchGapDetector()
        self.all_docs = all_docs or []

    def _normalize_score(self, score):
        try:
            score = float(score)
        except Exception:
            return 0.0

        if score < 0:
            score = 1.0 / (1.0 + abs(score))

        return max(0.0, min(score, 1.0))

    def _rank_chunks(self, question, chunks, intent):
        if not chunks:
            return []

        broad = intent == "broad_qa"
        ranked = []

        broad_terms = [
            "survey",
            "review",
            "overview",
            "introduction",
            "foundations",
            "fundamentals",
            "tutorial",
            "guide",
            "baseline",
        ]

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
                q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
                t_tokens = set(re.findall(r"[a-z0-9]+", title))
                overlap = len(q_tokens & t_tokens)
                bonus += min(0.20, overlap * 0.03)

            final_score = min(1.0, base_score + bonus)

            item = dict(c)
            item["score"] = final_score
            ranked.append(item)

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def _confidence(self, chunks):
        if not chunks:
            return "low", 0.0

        scores = []
        for c in chunks[:5]:
            try:
                score = self._normalize_score(c.get("score", 0.0))
                scores.append(score)
            except Exception:
                continue

        if not scores:
            return "low", 0.0

        avg_score = sum(scores) / len(scores)

        if avg_score >= 0.75:
            return "high", round(avg_score, 4)
        if avg_score >= 0.45:
            return "medium", round(avg_score, 4)
        return "low", round(avg_score, 4)

    def _build_prompt(self, question, chunks):
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

    def _fallback_answer(self, question, chunks):
        if not chunks:
            return "I could not find enough evidence to answer reliably."

        lines = []
        for c in chunks[:3]:
            title = c.get("title", "Unknown title")
            excerpt = c.get("text", "")[:240].strip()
            lines.append(f"- {title}: {excerpt}...")

        return "Based on the best available evidence:\n" + "\n".join(lines)

    def _compare_answer(self, question, entities, domain=None):
        if len(entities) < 2:
            return None

        left_name, right_name = entities[0], entities[1]

        left_chunks = self.retriever.search(
            left_name,
            top_k=7,
            domain=domain,
            must_contain_terms=[left_name]
        )

        right_chunks = self.retriever.search(
            right_name,
            top_k=7,
            domain=domain,
            must_contain_terms=[right_name]
        )

        if not left_chunks:
            fallback_left = self.retriever.search(left_name, top_k=7, domain=domain)
            left_chunks = [
                c for c in fallback_left
                if left_name.lower() in f"{c.get('title', '')} {c.get('text', '')}".lower()
            ]

        if not right_chunks:
            fallback_right = self.retriever.search(right_name, top_k=7, domain=domain)
            right_chunks = [
                c for c in fallback_right
                if right_name.lower() in f"{c.get('title', '')} {c.get('text', '')}".lower()
            ]

        if not left_chunks or not right_chunks:
            answer = (
                f"I could not find enough direct evidence in the current corpus to compare "
                f"{left_name} and {right_name} reliably. The compare mode is now strict enough "
                f"to avoid unrelated drift, so it will only answer when both entities appear directly."
            )

            return {
                "question": question,
                "intent": "compare",
                "rewritten_query": f"compare {left_name} and {right_name}",
                "answer": answer,
                "confidence": "low",
                "confidence_score": 0.10,
                "sources": [],
                "verification": {
                    "evidence": {
                        "label": "risky",
                        "status": "weak",
                        "support_score": 0.0,
                        "issues": ["Not enough direct evidence for both compare targets."],
                        "unsupported_sentences": [answer[:250]],
                        "source_coverage": 0
                    },
                    "contradiction": {
                        "label": "consistent",
                        "status": "clean",
                        "contradiction_score": 0.0,
                        "issues": ["No contradiction analysis needed because direct evidence was not found."],
                        "conflicting_pairs": []
                    }
                },
                "processing_time": 0.0,
                "compare_entities": {
                    "left": left_name,
                    "right": right_name,
                    "left_confidence": "low",
                    "right_confidence": "low",
                }
            }
    def answer(self, question, domain=None, top_k=5):
        start = time.time()
        plan = self.router.build_plan(question, domain=domain)

        if plan.intent == "compare" and len(plan.entities) >= 2:
            compare_result = self._compare_answer(question, plan.entities, domain=plan.domain)
            if compare_result is not None:
                compare_result["processing_time"] = round(time.time() - start, 4)
                compare_result["citations"] = self.citations.export_bibliography(compare_result["sources"])
                compare_result["research_gaps"] = self.gap_detector.detect_gaps(self.all_docs, topic=question)
                return compare_result

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
        else:
            chunks = self.retriever.search(
                plan.rewritten_query,
                top_k=plan.top_k,
                domain=plan.domain
            )

        chunks = self._rank_chunks(question, chunks, plan.intent)

        confidence_label, confidence_score = self._confidence(chunks)

        prompt = self._build_prompt(question, chunks)

        if self.llm_fn:
            try:
                answer_text = self.llm_fn(prompt)
            except Exception:
                answer_text = self._fallback_answer(question, chunks)
        else:
            answer_text = self._fallback_answer(question, chunks)

        sources = []
        for c in chunks[:3]:
            sources.append({
                "title": c.get("title"),
                "year": c.get("year"),
                "source": c.get("source"),
                "url": c.get("url"),
                "score": round(float(c.get("score", 0.0)), 4),
                "excerpt": c.get("text", "")[:300],
            })

        evidence_verification = self.evidence_checker.safety_label(answer_text, sources)
        contradiction_verification = self.contradiction_checker.safety_label(
            answer_text,
            sources,
            question=question,
            intent=plan.intent
        )
        if contradiction_verification["label"] == "conflicted":
            confidence_label = "low"
            confidence_score = min(confidence_score, 0.25)

        if evidence_verification["label"] == "risky":
            confidence_label = "low"
            confidence_score = min(confidence_score, 0.25)

        return {
            "question": question,
            "intent": plan.intent,
            "rewritten_query": plan.rewritten_query,
            "answer": answer_text,
            "confidence": confidence_label,
            "confidence_score": confidence_score,
            "sources": sources,
            "citations": self.citations.export_bibliography(sources),
            "verification": {
                "evidence": evidence_verification,
                "contradiction": contradiction_verification,
            },
            "research_gaps": self.gap_detector.detect_gaps(self.all_docs, topic=question),
            "processing_time": round(time.time() - start, 4),
        }

    def summarize_topic(self, topic, domain=None):
        q = f"Write a short literature summary on {topic} with methods, findings, and limitations."
        return self.answer(q, domain=domain)

    def compare_topics(self, topic_a, topic_b, domain=None):
        q = f"Compare {topic_a} and {topic_b}"
        return self.answer(q, domain=domain)