import re
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class QueryPlan:
    intent: str
    domain: Optional[str]
    rewritten_query: str
    top_k: int
    entities: List[str]


class QueryRouter:
    def __init__(self):
        self.broad_patterns = [
            r"\bwhat is\b",
            r"\bdefine\b",
            r"\bdefinition\b",
            r"\boverview\b",
            r"\bintroduction\b",
            r"\bbasics\b",
            r"\bfundamentals\b",
            r"\bexplain\b",
        ]

        self.compare_patterns = [
            r"\bcompare\b",
            r"\bdifference between\b",
            r"\bvs\b",
            r"\bversus\b",
        ]

        self.summary_patterns = [
            r"\bsummarize\b",
            r"\bsummary\b",
            r"\bliterature review\b",
        ]

    def detect_intent(self, question: str) -> str:
        q = question.lower()

        if any(re.search(pat, q) for pat in self.compare_patterns):
            return "compare"

        if any(re.search(pat, q) for pat in self.summary_patterns):
            return "summarize"

        if any(re.search(pat, q) for pat in self.broad_patterns):
            return "broad_qa"

        return "specific_qa"

    def extract_compare_entities(self, question: str) -> List[str]:
        """
        Try to extract the two things being compared.
        Examples:
        - Compare BERT and RoBERTa
        - BERT vs RoBERTa
        - Difference between CNN and Transformer
        """
        q = question.strip()

        # Remove common compare words first
        q = re.sub(r"(?i)\bcompare\b", "", q)
        q = re.sub(r"(?i)\bdifference between\b", "", q)

        # Split on vs / versus / and
        parts = re.split(r"(?i)\bvs\b|\bversus\b|\band\b", q)

        entities = []
        for part in parts:
            cleaned = part.strip(" ,:-;?")
            if cleaned:
                entities.append(cleaned)

        # Keep only the first 2 for compare mode
        return entities[:2]

    def rewrite_query(self, question: str, intent: str, entities: Optional[List[str]] = None) -> str:
        q = question.strip()

        if intent == "broad_qa":
            return f"foundational overview basics survey {q}"

        if intent == "compare" and entities and len(entities) >= 2:
            return f"{entities[0]} {entities[1]} comparison differences strengths weaknesses"

        if intent == "summarize":
            return f"literature review summary methods findings limitations {q}"

        return q

    def build_plan(self, question: str, domain: Optional[str] = None) -> QueryPlan:
        intent = self.detect_intent(question)

        if intent == "compare":
            entities = self.extract_compare_entities(question)
            top_k = 6
            rewritten_query = self.rewrite_query(question, intent, entities)

        elif intent == "broad_qa":
            entities = []
            top_k = 12
            rewritten_query = self.rewrite_query(question, intent)

        elif intent == "summarize":
            entities = []
            top_k = 8
            rewritten_query = self.rewrite_query(question, intent)

        else:
            entities = []
            top_k = 6
            rewritten_query = self.rewrite_query(question, intent)

        return QueryPlan(
            intent=intent,
            domain=domain,
            rewritten_query=rewritten_query,
            top_k=top_k,
            entities=entities,
        )