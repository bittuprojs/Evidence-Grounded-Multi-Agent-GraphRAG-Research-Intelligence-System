import re
from typing import List, Optional

from indexing.chunker import chunk_text


class PaperWriter:
    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _call_llm(self, prompt: str) -> str:
        if self.llm_fn:
            try:
                return self.llm_fn(prompt)
            except Exception as e:
                return f"LLM generation failed: {e}"
        return "LLM is not configured."

    def _select_relevant_chunks(self, question: str, paper_text: str, top_n: int = 4) -> List[str]:
        chunks = chunk_text(paper_text, chunk_size=220, overlap=40)
        if not chunks:
            return []

        q_tokens = set(self._tokenize(question))
        scored = []

        for chunk in chunks:
            c_tokens = set(self._tokenize(chunk))
            overlap = len(q_tokens & c_tokens)
            score = overlap / max(len(q_tokens), 1)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_n]]

    def analyze_paper(self, paper_text: str, title: str = "Uploaded paper") -> str:
        prompt = f"""
You are an expert academic research assistant.

Analyze the paper titled: {title}

Return the result in this structure:

1. Summary
2. Main contribution
3. Methodology
4. Strengths
5. Limitations
6. Future work
7. Keywords
8. Suggested citation style

Paper:
{paper_text[:12000]}
"""
        return self._call_llm(prompt)

    def summarize_paper(self, paper_text: str, title: str = "Uploaded paper") -> str:
        prompt = f"""
Write a clean academic summary of this research paper.

Title: {title}

Return:
- 1 paragraph summary
- main contribution
- method
- limitations
- future work

Paper:
{paper_text[:12000]}
"""
        return self._call_llm(prompt)

    def answer_about_paper(self, question: str, paper_text: str, title: str = "Uploaded paper") -> str:
        relevant = self._select_relevant_chunks(question, paper_text, top_n=5)
        evidence = "\n\n".join([f"[Section {i+1}]\n{c}" for i, c in enumerate(relevant)])

        prompt = f"""
You are answering a question only from the uploaded research paper.

Paper title: {title}

Question: {question}

Use only the evidence below. If the evidence is insufficient, say so.

Evidence:
{evidence}

Return:
- concise answer
- key evidence points
- any limitations
"""
        return self._call_llm(prompt)

    def generate_outline(self, topic: str, goal: str = "research paper") -> str:
        prompt = f"""
Create a strong academic outline for a {goal} on the topic:

{topic}

Include:
1. Title
2. Abstract idea
3. Introduction
4. Related Work
5. Methodology
6. Experiments / Evaluation
7. Results
8. Limitations
9. Future Work
10. Conclusion
"""
        return self._call_llm(prompt)

    def draft_literature_review(self, topic: str, evidence_text: Optional[str] = None) -> str:
        prompt = f"""
Write a literature review section for a research paper on:

{topic}

Requirements:
- academic tone
- summarize trends
- compare approaches
- mention limitations
- mention research gaps
- keep it structured and useful for a final-year project

Evidence:
{(evidence_text or '')[:12000]}
"""
        return self._call_llm(prompt)

    def improve_paragraph(self, paragraph: str) -> str:
        prompt = f"""
Rewrite the following paragraph in a clear academic style suitable for a research paper.
Keep the meaning the same, improve clarity, grammar, and flow.

Paragraph:
{paragraph}
"""
        return self._call_llm(prompt)

    def compare_uploaded_vs_topic(self, paper_text: str, uploaded_title: str, topic: str) -> str:
        relevant = self._select_relevant_chunks(topic, paper_text, top_n=5)
        evidence = "\n\n".join([f"[Section {i+1}]\n{c}" for i, c in enumerate(relevant)])

        prompt = f"""
You are comparing an uploaded paper to a topic.

Uploaded paper title: {uploaded_title}
Comparison topic: {topic}

Return:
1. Similarities
2. Differences
3. What the uploaded paper contributes
4. What the topic adds beyond the paper
5. Research gap or next step
6. Short final verdict

Use only the evidence below from the uploaded paper:

Evidence:
{evidence}
"""
        return self._call_llm(prompt)