from rank_bm25 import BM25Okapi
from retrieval.reranker import Reranker


class HybridRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.reranker = Reranker()
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None

    def fit(self, chunks):
        self.chunks = chunks
        self.tokenized_corpus = [c["text"].lower().split() for c in chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _build_where(self, domain=None, source=None):
        where = {}

        if domain not in [None, "", "General", "All Domains"]:
            where["domain"] = domain

        if source not in [None, "", "All"]:
            where["source"] = source

        if not where:
            return None

        return where

    def _chunk_matches_terms(self, chunk, terms):
        if not terms:
            return True

        text = f"{chunk.get('title', '')} {chunk.get('text', '')}".lower()
        for term in terms:
            if term and term.lower() in text:
                return True
        return False

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain: str | None = None,
        source: str | None = None,
        must_contain_terms: list[str] | None = None
    ):
        """
        Hybrid search:
        - vector retrieval
        - BM25 keyword retrieval
        - reranking
        - optional domain/source filtering
        - optional strict term filter for compare mode
        """
        where = self._build_where(domain=domain, source=source)

        vector_hits = self.vector_store.query(query, top_k=top_k * 3, where=where)

        bm25_hits = []
        if self.bm25 is not None and self.chunks:
            q_tokens = query.lower().split()
            scores = self.bm25.get_scores(q_tokens)
            ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)

            for chunk, score in ranked[:top_k * 3]:
                if domain not in [None, "", "General", "All Domains"]:
                    if chunk.get("domain") != domain:
                        continue

                if source not in [None, "", "All"]:
                    if chunk.get("source") != source:
                        continue

                item = dict(chunk)
                item["score"] = float(max(score, 0.0))
                bm25_hits.append(item)

        merged = {}

        for c in vector_hits:
            merged[c["chunk_id"]] = c

        for c in bm25_hits:
            if c["chunk_id"] in merged:
                merged[c["chunk_id"]]["score"] = max(
                    merged[c["chunk_id"]]["score"],
                    c["score"]
                )
            else:
                merged[c["chunk_id"]] = c

        merged_list = list(merged.values())
        merged_list.sort(key=lambda x: x["score"], reverse=True)

        if must_contain_terms:
            strict = [
                c for c in merged_list
                if self._chunk_matches_terms(c, must_contain_terms)
            ]
            if not strict:
                return []
            merged_list = strict

        reranked = self.reranker.rerank(query, merged_list, top_k=top_k)
        return reranked