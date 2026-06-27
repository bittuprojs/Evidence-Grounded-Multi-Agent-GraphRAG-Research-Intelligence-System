import json
from pathlib import Path

from analytics.graph_builder import ResearchGraph
from generation.answer_engine import AnswerEngine
from generation.llm_client import GeminiLLM
from indexing.chunker import build_chunks
from indexing.vector_store import VectorStore
from retrieval.hybrid_retriever import HybridRetriever

BASE_DIR = Path(__file__).resolve().parent

FOUNDATION_FILE = BASE_DIR / "data" / "processed" / "papers_with_foundations.json"
PAPERS_FILE = FOUNDATION_FILE if FOUNDATION_FILE.exists() else (BASE_DIR / "data" / "processed" / "papers.json")
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"


def load_documents():
    if not PAPERS_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {PAPERS_FILE}")

    with open(PAPERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chunks(chunks):
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def build_system():
    docs = load_documents()

    print("=" * 60)
    print("BUILDING CHUNKS")
    print("=" * 60)
    chunks = build_chunks(docs, chunk_size=180, overlap=40)
    print(f"Created chunks: {len(chunks)}")
    save_chunks(chunks)

    print("=" * 60)
    print("INITIALIZING VECTOR STORE")
    print("=" * 60)
    vector_store = VectorStore(collection_name="research_papers_v6")
    vector_store.upsert_chunks(chunks)
    print(f"Indexed chunks: {vector_store.count()}")

    print("=" * 60)
    print("INITIALIZING RETRIEVER")
    print("=" * 60)
    retriever = HybridRetriever(vector_store)
    retriever.fit(chunks)

    print("=" * 60)
    print("INITIALIZING GEMINI")
    print("=" * 60)
    try:
        llm = GeminiLLM()
        llm_fn = llm.generate
    except Exception as e:
        print(f"Gemini unavailable, using fallback only: {e}")
        llm_fn = None

    print("=" * 60)
    print("INITIALIZING ANSWER ENGINE")
    print("=" * 60)
    answer_engine = AnswerEngine(retriever=retriever, llm_fn=llm_fn, all_docs=docs)

    print("=" * 60)
    print("BUILDING ANALYTICS")
    print("=" * 60)
    graph = ResearchGraph()
    graph.build_from_documents(docs)

    return answer_engine, docs, graph