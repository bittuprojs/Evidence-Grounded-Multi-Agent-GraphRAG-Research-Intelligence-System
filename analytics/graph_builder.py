from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional
import re

import networkx as nx
import matplotlib.pyplot as plt


class ResearchGraph:
    def __init__(self):
        self.paper_nodes = {}
        self.topic_index = defaultdict(set)
        self.author_index = defaultdict(set)
        self.domain_index = defaultdict(set)
        self.documents = []

        self.stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "are", "was",
            "were", "into", "using", "use", "based", "paper", "study", "data",
            "model", "models", "method", "methods", "results", "analysis",
            "their", "there", "been", "have", "has", "had", "also", "such",
            "than", "then", "when", "where", "while", "about", "these", "those",
            "them", "they", "its", "it", "an", "a", "of", "to", "in", "on",
            "by", "as", "at", "be", "is", "or", "we", "our", "this"
        }

    def _tokenize(self, text: str):
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if len(t) > 3 and t not in self.stopwords]

    def add_document(self, doc: Dict[str, Any]):
        doc_id = doc.get("doc_id")
        if not doc_id:
            return

        self.paper_nodes[doc_id] = {
            "doc_id": doc_id,
            "title": doc.get("title", ""),
            "domain": doc.get("domain", "General"),
            "year": doc.get("year"),
            "authors": doc.get("authors", []),
            "url": doc.get("url"),
            "source": doc.get("source", "unknown"),
            "text": doc.get("text", "") or doc.get("full_text", "")
        }

        domain = doc.get("domain", "General")
        self.domain_index[domain].add(doc_id)

        title = (doc.get("title", "") or "").lower()
        text = (doc.get("text", "") or doc.get("full_text", "") or "").lower()

        for token in self._tokenize(title + " " + text[:700]):
            self.topic_index[token].add(doc_id)

        for author in doc.get("authors", []):
            self.author_index[str(author).lower()].add(doc_id)

    def build_from_documents(self, docs: List[Dict[str, Any]]):
        self.documents = docs
        self.paper_nodes = {}
        self.topic_index = defaultdict(set)
        self.author_index = defaultdict(set)
        self.domain_index = defaultdict(set)

        for doc in docs:
            self.add_document(doc)

    def domain_summary(self) -> Dict[str, int]:
        return {domain: len(ids) for domain, ids in self.domain_index.items()}

    def top_keywords(self, top_k: int = 15) -> List[tuple]:
        freq = Counter()

        for doc in self.documents:
            text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
            for token in self._tokenize(text):
                freq[token] += 1

        return freq.most_common(top_k)

    def _score_doc_for_topic(self, doc: Dict[str, Any], topic: str) -> int:
        if not topic:
            return 1

        topic_tokens = set(self._tokenize(topic))
        text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
        doc_tokens = set(self._tokenize(text))

        score = len(topic_tokens & doc_tokens)

        title = (doc.get("title", "") or "").lower()
        for t in topic_tokens:
            if t in title:
                score += 2

        return score

    def select_relevant_docs(self, topic: str = "", max_papers: int = 15) -> List[Dict[str, Any]]:
        if not self.documents:
            return []

        if not topic.strip():
            # default: take papers with broadest keyword variety
            scored = []
            for doc in self.documents:
                score = len(self._tokenize(doc.get("title", "") + " " + doc.get("text", "")[:500]))
                scored.append((score, doc))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[:max_papers]]

        scored = []
        for doc in self.documents:
            score = self._score_doc_for_topic(doc, topic)
            if score > 0:
                scored.append((score, doc))

        if not scored:
            # fallback to broad docs
            for doc in self.documents:
                score = len(self._tokenize(doc.get("title", "") + " " + doc.get("text", "")[:500]))
                scored.append((score, doc))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[:max_papers]]

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:max_papers]]

    def build_topic_graph(self, topic: str = "", max_papers: int = 15, max_keywords: int = 10) -> nx.Graph:
        """
        Build a small, readable graph for dashboard display.
        Nodes:
        - topic node
        - domain nodes
        - paper nodes
        - keyword nodes
        """
        G = nx.Graph()
        selected_docs = self.select_relevant_docs(topic, max_papers=max_papers)

        topic_node = None
        if topic.strip():
            topic_node = f"topic::{topic.strip()}"
            G.add_node(topic_node, node_type="topic", label=topic.strip())

        # add domain nodes
        for doc in selected_docs:
            domain = doc.get("domain", "General")
            domain_node = f"domain::{domain}"
            if not G.has_node(domain_node):
                G.add_node(domain_node, node_type="domain", label=domain)

        # top keywords from selected docs
        local_freq = Counter()
        for doc in selected_docs:
            text = f"{doc.get('title', '')} {doc.get('text', '')[:800]}".lower()
            for tok in self._tokenize(text):
                local_freq[tok] += 1

        keywords = [kw for kw, _ in local_freq.most_common(max_keywords)]

        for kw in keywords:
            kw_node = f"kw::{kw}"
            G.add_node(kw_node, node_type="keyword", label=kw)

        # add paper nodes and edges
        for doc in selected_docs:
            doc_id = doc["doc_id"]
            title = doc.get("title", "Untitled")
            domain = doc.get("domain", "General")
            year = doc.get("year", "")

            paper_node = f"paper::{doc_id}"
            paper_label = title if len(title) <= 42 else title[:39] + "..."
            G.add_node(
                paper_node,
                node_type="paper",
                label=paper_label,
                title=title,
                year=year,
                domain=domain
            )

            # connect to domain
            G.add_edge(paper_node, f"domain::{domain}")

            # connect to topic
            if topic_node:
                G.add_edge(paper_node, topic_node)

            # connect to keywords
            doc_text = f"{doc.get('title', '')} {doc.get('text', '')[:800]}".lower()
            doc_tokens = set(self._tokenize(doc_text))
            for kw in keywords:
                if kw in doc_tokens:
                    G.add_edge(paper_node, f"kw::{kw}")

        # connect topic to keywords if topic terms appear
        if topic_node:
            topic_tokens = set(self._tokenize(topic))
            for kw in keywords:
                if kw in topic_tokens:
                    G.add_edge(topic_node, f"kw::{kw}")

        return G

    def plot_topic_graph(self, topic: str = "", max_papers: int = 15, max_keywords: int = 10):
        """
        Return a matplotlib figure for Streamlit.
        """
        G = self.build_topic_graph(topic=topic, max_papers=max_papers, max_keywords=max_keywords)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        if G.number_of_nodes() == 0:
            ax.set_title("No graph data available")
            ax.axis("off")
            return fig

        pos = nx.spring_layout(G, seed=42, k=0.9)

        node_types = nx.get_node_attributes(G, "node_type")

        paper_nodes = [n for n, t in node_types.items() if t == "paper"]
        domain_nodes = [n for n, t in node_types.items() if t == "domain"]
        keyword_nodes = [n for n, t in node_types.items() if t == "keyword"]
        topic_nodes = [n for n, t in node_types.items() if t == "topic"]

        # draw edges first
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=1.0)

        # draw nodes by type
        if domain_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=domain_nodes,
                node_size=900,
                ax=ax,
                alpha=0.95
            )

        if keyword_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=keyword_nodes,
                node_size=650,
                ax=ax,
                alpha=0.85
            )

        if paper_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=paper_nodes,
                node_size=500,
                ax=ax,
                alpha=0.85
            )

        if topic_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=topic_nodes,
                node_size=1400,
                ax=ax,
                alpha=1.0
            )

        labels = nx.get_node_attributes(G, "label")
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

        title = "Research Graph"
        if topic.strip():
            title = f"Research Graph for: {topic.strip()}"
        ax.set_title(title)
        ax.axis("off")

        fig.tight_layout()
        return fig