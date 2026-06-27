import re
from typing import List, Dict, Optional


class ContradictionChecker:
    def __init__(self):
        self.stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "are", "was",
            "were", "into", "using", "use", "based", "paper", "study", "data",
            "model", "models", "method", "methods", "results", "analysis",
            "their", "there", "been", "have", "has", "had", "also", "such",
            "than", "then", "when", "where", "while", "about", "this", "these",
            "those", "them", "they", "its", "it", "an", "a", "of", "to", "in"
        }

        self.opposites = [
            ("increase", "decrease"),
            ("improve", "worsen"),
            ("higher", "lower"),
            ("more", "less"),
            ("better", "worse"),
            ("support", "oppose"),
            ("allow", "prevent"),
            ("effective", "ineffective"),
            ("accurate", "inaccurate"),
            ("strong", "weak"),
            ("robust", "fragile"),
            ("positive", "negative"),
            ("present", "absent"),
            ("true", "false"),
            ("useful", "useless"),
            ("successful", "unsuccessful"),
        ]

        self.negation_words = {
            "no", "not", "never", "none", "without", "cannot", "can't", "wont", "won't", "n't"
        }

        self.broad_markers = [
            "what is", "define", "definition", "overview", "introduction",
            "basics", "fundamentals", "explain", "machine learning",
            "deep learning", "retrieval augmented generation", "rag"
        ]

    def _tokenize(self, text: str):
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if len(t) > 2 and t not in self.stopwords]

    def _sentence_split(self, text: str):
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _has_negation(self, text: str) -> bool:
        t = text.lower()
        return any(word in t for word in self.negation_words)

    def _shared_content_score(self, a: str, b: str) -> float:
        a_tokens = set(self._tokenize(a))
        b_tokens = set(self._tokenize(b))
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / max(min(len(a_tokens), len(b_tokens)), 1)

    def _opposite_term_match(self, a: str, b: str) -> bool:
        a_l = a.lower()
        b_l = b.lower()

        for x, y in self.opposites:
            if (x in a_l and y in b_l) or (y in a_l and x in b_l):
                return True
        return False

    def _is_broad_context(self, question: Optional[str], answer: Optional[str]) -> bool:
        text = f"{question or ''} {answer or ''}".lower()
        return any(marker in text for marker in self.broad_markers)

    def detect(
        self,
        answer: str,
        sources: List[Dict],
        question: Optional[str] = None,
        intent: Optional[str] = None
    ) -> Dict:
        """
        Heuristic contradiction detector.
        For broad foundational answers, it is intentionally relaxed to avoid false conflicts.
        """
        if not answer or not sources:
            return {
                "status": "unknown",
                "contradiction_score": 0.0,
                "issues": [],
                "conflicting_pairs": []
            }

        is_broad = intent == "broad_qa" or self._is_broad_context(question, answer)

        # Relax contradiction checks for broad definition-style answers.
        if is_broad and len(sources) <= 5:
            return {
                "status": "clean",
                "contradiction_score": 0.0,
                "issues": ["Broad conceptual answer; contradiction checks relaxed."],
                "conflicting_pairs": []
            }

        conflicts = []
        source_texts = [
            f"{s.get('title', '')} {s.get('excerpt', '')}".strip()
            for s in sources
        ]

        answer_sentences = self._sentence_split(answer)
        for sent in answer_sentences:
            best_match = None
            best_score = 0.0

            for idx, src_text in enumerate(source_texts):
                score = self._shared_content_score(sent, src_text)
                if score > best_score:
                    best_score = score
                    best_match = idx

            if best_match is None or best_score < 0.18:
                continue

            src_text = source_texts[best_match]
            sent_neg = self._has_negation(sent)
            src_neg = self._has_negation(src_text)

            if sent_neg != src_neg and best_score >= 0.25:
                conflicts.append({
                    "type": "negation_mismatch",
                    "answer_sentence": sent,
                    "source_title": sources[best_match].get("title", "Unknown"),
                    "source_excerpt": sources[best_match].get("excerpt", "")[:220],
                    "score": round(best_score, 4),
                })

            if self._opposite_term_match(sent, src_text) and best_score >= 0.22:
                conflicts.append({
                    "type": "opposite_terms",
                    "answer_sentence": sent,
                    "source_title": sources[best_match].get("title", "Unknown"),
                    "source_excerpt": sources[best_match].get("excerpt", "")[:220],
                    "score": round(best_score, 4),
                })

        for i in range(len(source_texts)):
            for j in range(i + 1, len(source_texts)):
                a = source_texts[i]
                b = source_texts[j]

                shared = self._shared_content_score(a, b)
                if shared < 0.20:
                    continue

                a_neg = self._has_negation(a)
                b_neg = self._has_negation(b)

                if a_neg != b_neg and shared >= 0.25:
                    conflicts.append({
                        "type": "source_negation_conflict",
                        "source_a": sources[i].get("title", "Unknown"),
                        "source_b": sources[j].get("title", "Unknown"),
                        "score": round(shared, 4),
                    })

                if self._opposite_term_match(a, b) and shared >= 0.25:
                    conflicts.append({
                        "type": "source_opposite_terms",
                        "source_a": sources[i].get("title", "Unknown"),
                        "source_b": sources[j].get("title", "Unknown"),
                        "score": round(shared, 4),
                    })

        if conflicts:
            avg_score = sum(c["score"] for c in conflicts) / len(conflicts)
            status = "conflict" if avg_score >= 0.25 else "watch"
        else:
            avg_score = 0.0
            status = "clean"

        issues = []
        if status == "conflict":
            issues.append("Possible contradiction found across answer or sources.")
        elif status == "watch":
            issues.append("Some potentially conflicting evidence was detected.")
        else:
            issues.append("No obvious contradiction detected.")

        return {
            "status": status,
            "contradiction_score": round(avg_score, 4),
            "issues": issues,
            "conflicting_pairs": conflicts[:8]
        }

    def safety_label(
        self,
        answer: str,
        sources: List[Dict],
        question: Optional[str] = None,
        intent: Optional[str] = None
    ) -> Dict:
        result = self.detect(answer, sources, question=question, intent=intent)

        if result["status"] == "clean":
            label = "consistent"
        elif result["status"] == "watch":
            label = "uncertain"
        else:
            label = "conflicted"

        return {
            "label": label,
            **result
        }