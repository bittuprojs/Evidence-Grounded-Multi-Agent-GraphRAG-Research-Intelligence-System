from typing import List, Dict
from ingestion.cleaner import clean_text


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> List[str]:
    """
    Split text into overlapping word chunks.
    """
    text = clean_text(text)

    if not text:
        return []

    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_chunks(docs: List[Dict], chunk_size: int = 180, overlap: int = 40) -> List[Dict]:
    all_chunks = []

    for doc in docs:
        text = doc.get("text", "") or doc.get("full_text", "")
        pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for idx, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f'{doc["doc_id"]}_chunk_{idx}',
                "doc_id": doc["doc_id"],
                "title": doc.get("title"),
                "text": piece,
                "source": doc.get("source"),
                "domain": doc.get("domain"),
                "authors": doc.get("authors", []),
                "year": doc.get("year"),
                "url": doc.get("url"),
                "metadata": doc.get("metadata", {})
            })

    return all_chunks