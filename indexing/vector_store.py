import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, EMBED_MODEL


class VectorStore:

    def __init__(self, collection_name="research_papers"):

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        self.model = SentenceTransformer(EMBED_MODEL)

    def count(self):
        return self.collection.count()

    def upsert_chunks(self, chunks):

        if not chunks:
            return

        ids = []
        docs = []
        metas = []
        embeddings = []

        batch_size = 128

        for i in range(0, len(chunks), batch_size):

            batch = chunks[i:i + batch_size]

            batch_ids = [c["chunk_id"] for c in batch]
            batch_docs = [c["text"] for c in batch]

            batch_metas = []

            for c in batch:

                batch_metas.append({
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "title": c.get("title"),
                    "source": c.get("source"),
                    "domain": c.get("domain"),
                    "year": c.get("year"),
                    "url": c.get("url")
                })

            batch_embeddings = self.model.encode(
                batch_docs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            ).tolist()

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embeddings
            )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where=None
    ):

        q_emb = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        result = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        hits = []

        for doc, meta, dist in zip(docs, metas, dists):

            similarity = 1.0 - float(dist)

            hits.append({
                "chunk_id": meta.get("chunk_id"),
                "doc_id": meta.get("doc_id"),
                "title": meta.get("title"),
                "text": doc,
                "score": max(0.0, similarity),
                "source": meta.get("source"),
                "domain": meta.get("domain"),
                "year": meta.get("year"),
                "url": meta.get("url"),
                "metadata": meta
            })

        return hits