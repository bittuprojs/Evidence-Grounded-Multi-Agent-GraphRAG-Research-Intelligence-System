import re
import hashlib
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove obvious noise.
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_doc_key(title: str, text: str) -> str:
    """
    Create a stable hash for duplicate detection.

    Using only the first 200 characters avoids over-collapsing
    papers that start similarly but are actually different.
    """
    title_part = (title or "").strip().lower()
    text_part = (text or "").strip().lower()[:200]
    base = f"{title_part}||{text_part}".encode("utf-8")
    return hashlib.md5(base).hexdigest()


def deduplicate_documents(docs: List[Dict]) -> List[Dict]:
    """
    Remove duplicate documents using a stable content hash.
    """
    seen = set()
    clean_docs = []

    for doc in docs:
        title = doc.get("title", "")
        text = doc.get("text", "") or doc.get("abstract", "")

        if not text.strip():
            continue

        key = make_doc_key(title, text)
        if key in seen:
            continue

        seen.add(key)
        doc["text"] = clean_text(text)
        clean_docs.append(doc)

    return clean_docs


def filter_short_docs(docs: List[Dict], min_chars: int = 200) -> List[Dict]:
    """
    Drop documents that are too short to be useful.
    """
    return [d for d in docs if len(d.get("text", "")) >= min_chars]