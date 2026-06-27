import re
from typing import List, Dict, Optional


class EvidenceChecker:
    def __init__(self):
        self.stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "are", "was",
            "were", "into", "using", "use", "based", "paper", "study", "data",
            "model", "models", "method", "methods", "results", "analysis",
            "their", "there", "been", "have", "has", "had", "also", "such",
            "than", "then", "when", "where", "while", "about", "these", "those",
            "them", "they", "its", "it", "an", "a", "of", "to", "in", "on",
            "by", "as", "at", "be", "is", "or", "we", "our", "not", "no"
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

    def _best_sentence_support(self, sentence: str, sources: List[Dict]) -> float:
        sent_tokens = set(self._tokenize(sentence))
        if not sent_tokens:
            return 0.0

        best = 0.0
        for src in sources:
            src_text = f"{src.get('excerpt', '')} {src.get('title', '')}".lower()
            src_tokens = set(self._tokenize(src_text))
            if not src_tokens:
                continue

            overlap = len(sent_tokens & src_tokens)
            score = overlap / max(len(sent_tokens), 1)
            if score > best:
                best = score

        return best

    def _is_broad_context(self, question: Optional[str], answer: Optional[str], intent: Optional[str]) -> bool:
        if intent == "broad_qa":
            return True

        text = f"{question or ''} {answer or ''}".lower()
        return any(marker in text for marker in self.broad_markers)

    def score_support(
        self,
        answer: str,
        sources: List[Dict],
        question: Optional[str] = None,
        intent: Optional[str] = None
    ) -> Dict:
        if not answer:
            return {
                "status": "weak",
                "support_score": 0.0,
                "issues": ["Empty answer text."],
                "unsupported_sentences": [],
                "source_coverage": 0
            }

        if not sources:
            return {
                "status": "weak",
                "support_score": 0.0,
                "issues": ["No sources available."],
                "unsupported_sentences": [answer[:250]],
                "source_coverage": 0
            }

        is_broad = self._is_broad_context(question, answer, intent)

        answer_sents = self._sentence_split(answer)
        unsupported = []
        sentence_scores = []
        source_hits = 0

        for sent in answer_sents:
            score = self._best_sentence_support(sent, sources)

            # Relax support threshold for broad conceptual answers
            threshold = 0.10 if is_broad else 0.12

            if score > threshold:
                source_hits += 1
            else:
                unsupported.append(sent)

            sentence_scores.append(score)

        support_score = sum(sentence_scores) / max(len(sentence_scores), 1)

        unique_sources_supporting = 0
        for src in sources:
            src_text = f"{src.get('excerpt', '')} {src.get('title', '')}".lower()
            src_tokens = set(self._tokenize(src_text))
            if not src_tokens:
                continue

            for sent in answer_sents:
                sent_tokens = set(self._tokenize(sent))
                if not sent_tokens:
                    continue

                sent_overlap = len(sent_tokens & src_tokens) / max(len(sent_tokens), 1)
                if sent_overlap > (0.08 if is_broad else 0.10):
                    unique_sources_supporting += 1
                    break

        issues = []
        if support_score < 0.12 and not is_broad:
            issues.append("Overall evidence support is weak.")
        if len(unsupported) > 0:
            issues.append(f"{len(unsupported)} sentence(s) are weakly supported.")
        if len(sources) < 2:
            issues.append("Only one or very few sources retrieved.")
        if unique_sources_supporting < 2 and len(sources) >= 2:
            issues.append("Evidence is not sufficiently distributed across multiple sources.")

        if is_broad:
            # broad conceptual answers should not be punished too harshly
            if support_score >= 0.22 and len(unsupported) <= 2:
                status = "strong"
            elif support_score >= 0.14:
                status = "medium"
            else:
                status = "weak"
        else:
            if support_score >= 0.35 and len(unsupported) == 0:
                status = "strong"
            elif support_score >= 0.2:
                status = "medium"
            else:
                status = "weak"

        return {
            "status": status,
            "support_score": round(support_score, 4),
            "issues": issues,
            "unsupported_sentences": unsupported[:5],
            "source_coverage": unique_sources_supporting
        }

    def safety_label(
        self,
        answer: str,
        sources: List[Dict],
        question: Optional[str] = None,
        intent: Optional[str] = None
    ) -> Dict:
        result = self.score_support(answer, sources, question=question, intent=intent)

        if result["status"] == "strong":
            label = "supported"
        elif result["status"] == "medium":
            label = "caution"
        else:
            label = "risky"

        return {
            "label": label,
            **result
        }