import argparse
import json
from pathlib import Path
from generation.llm_client import GeminiLLM
from indexing.chunker import build_chunks
from indexing.vector_store import VectorStore
from retrieval.hybrid_retriever import HybridRetriever
from generation.answer_engine import AnswerEngine
from ui.streamlit_app import build_ui
from analytics.graph_builder import ResearchGraph
from analytics.trend import year_distribution, domain_distribution, top_keywords

BASE_DIR = Path(__file__).resolve().parent
FOUNDATION_PAPERS_FILE = BASE_DIR / "data" / "processed" / "papers_with_foundations.json"
PAPERS_FILE = FOUNDATION_PAPERS_FILE if FOUNDATION_PAPERS_FILE.exists() else (BASE_DIR / "data" / "processed" / "papers.json")
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


def run_pipeline():
    print("=" * 60)
    print("LOADING DOCUMENTS")
    print("=" * 60)

    docs = load_documents()
    print(f"Loaded documents: {len(docs)}")

    print("\n" + "=" * 60)
    print("BUILDING CHUNKS")
    print("=" * 60)

    chunks = build_chunks(docs, chunk_size=180, overlap=40)
    print(f"Created chunks: {len(chunks)}")
    save_chunks(chunks)

    print("\n" + "=" * 60)
    print("INITIALIZING VECTOR STORE")
    print("=" * 60)

    vector_store = VectorStore(collection_name="research_papers_v4")
    vector_store.upsert_chunks(chunks)
    print(f"Indexed chunks: {vector_store.count()}")

    print("\n" + "=" * 60)
    print("INITIALIZING RETRIEVER")
    print("=" * 60)

    retriever = HybridRetriever(vector_store)
    retriever.fit(chunks)

    print("\n" + "=" * 60)
    print("INITIALIZING ANSWER ENGINE")
    print("=" * 60)

    llm = GeminiLLM()
    answer_engine = AnswerEngine(retriever=retriever, llm_fn=llm.generate, all_docs=docs)

    print("\n" + "=" * 60)
    print("BUILDING ANALYTICS")
    print("=" * 60)

    graph = ResearchGraph()
    graph.build_from_documents(docs)

    print(f"Domain summary: {graph.domain_summary()}")
    print(f"Top keywords: {top_keywords(docs, top_k=10)}")

    return answer_engine, docs, graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--domain", type=str, default=None)
    args = parser.parse_args()

    answer_engine, docs, graph = run_pipeline()

    if args.question:
        result = answer_engine.answer(args.question, domain=args.domain)
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.ui:
        demo = build_ui(answer_engine,docs)
        demo.launch()


if __name__ == "__main__":
    main()