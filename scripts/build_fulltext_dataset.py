import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.pdf_loader import download_and_extract
from ingestion.cleaner import deduplicate_documents, filter_short_docs

BASE_DIR = ROOT
DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw" / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "papers.json"
OUTPUT_FILE = PROCESSED_DIR / "papers_fulltext.json"
STATS_FILE = PROCESSED_DIR / "fulltext_stats.json"

RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_docs():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )


def main():
    docs = load_docs()
    updated_docs = []

    downloaded = 0
    extracted = 0
    replaced_with_fulltext = 0

    print("=" * 60)
    print("BUILDING FULL-TEXT DATASET")
    print("=" * 60)
    print(f"Loaded documents: {len(docs)}")

    for idx, doc in enumerate(docs, 1):
        title = doc.get("title", f"doc_{idx}")
        pdf_url = doc.get("url") or doc.get("pdf_url")
        source = doc.get("source", "unknown")

        doc = dict(doc)

        # Default text is abstract / existing text
        base_text = doc.get("text", "") or doc.get("abstract", "")
        full_text = base_text

        # Try to fetch full PDF text for ArXiv documents
        if source == "arxiv" and pdf_url:
            pdf_name = f"{doc.get('doc_id', f'doc_{idx}')}.pdf"
            pdf_path = RAW_PDF_DIR / pdf_name

            print(f"[{idx}/{len(docs)}] Extracting: {title[:80]}")

            text = download_and_extract(pdf_url, pdf_path)
            if text:
                downloaded += 1
                extracted += 1

                # Use full text if it is meaningfully longer than the abstract
                if len(text) > len(base_text) + 500:
                    full_text = text
                    replaced_with_fulltext += 1

        doc["full_text"] = full_text
        doc["text"] = full_text

        updated_docs.append(doc)

    # Clean and deduplicate again
    updated_docs = deduplicate_documents(updated_docs)
    updated_docs = filter_short_docs(updated_docs, min_chars=200)

    save_json(OUTPUT_FILE, updated_docs)

    stats = {
        "input_documents": len(docs),
        "output_documents": len(updated_docs),
        "downloaded_pdfs": downloaded,
        "extracted_pdfs": extracted,
        "replaced_with_fulltext": replaced_with_fulltext,
    }
    save_json(STATS_FILE, stats)

    print("\n" + "=" * 60)
    print("FULL-TEXT DATASET COMPLETE")
    print("=" * 60)
    print(f"Output documents: {len(updated_docs)}")
    print(f"Downloaded PDFs: {downloaded}")
    print(f"Extracted PDFs: {extracted}")
    print(f"Replaced with full text: {replaced_with_fulltext}")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {STATS_FILE}")


if __name__ == "__main__":
    main()