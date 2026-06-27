try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.enabled = False
        self.model = None

        if CrossEncoder is not None:
            try:
                self.model = CrossEncoder(model_name)
                self.enabled = True
            except Exception:
                self.model = None
                self.enabled = False

    def rerank(self, query: str, chunks, top_k: int = 5):
        if not chunks:
            return []

        if self.enabled and self.model is not None:
            pairs = [(query, c["text"]) for c in chunks]
            scores = self.model.predict(pairs)
            ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            out = []
            for c, s in ranked[:top_k]:
                item = dict(c)
                item["score"] = float(s)
                out.append(item)
            return out

        # fallback rerank by overlap
        q_words = set(query.lower().split())
        scored = []
        for c in chunks:
            c_words = set(c["text"].lower().split())
            overlap = len(q_words & c_words)
            score = 0.7 * float(c["score"]) + 0.3 * (overlap / max(len(q_words), 1))
            item = dict(c)
            item["score"] = score
            scored.append(item)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]