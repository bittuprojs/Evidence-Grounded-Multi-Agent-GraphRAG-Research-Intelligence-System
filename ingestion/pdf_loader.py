from pathlib import Path
from typing import List, Dict
import requests

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


def download_pdf(url: str, save_path: Path, timeout: int = 60) -> bool:
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        headers = {"User-Agent": "Mozilla/5.0"}

        with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        return True

    except Exception as e:
        print(f"PDF download failed for {url}: {e}")
        return False


def extract_pdf_text(pdf_path: str) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        pages = []

        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
            except Exception:
                continue

        return "\n".join(pages).strip()

    except Exception as e:
        print(f"PDF extraction failed for {pdf_path}: {e}")
        return ""


def load_pdfs_from_folder(folder_path: str, domain: str = "General") -> List[Dict]:
    folder = Path(folder_path)
    docs = []

    if not folder.exists():
        print(f"PDF folder not found: {folder_path}")
        return docs

    for pdf_file in folder.glob("*.pdf"):
        try:
            text = extract_pdf_text(str(pdf_file))
            if not text.strip():
                continue

            docs.append({
                "doc_id": pdf_file.stem,
                "title": pdf_file.stem.replace("_", " "),
                "text": text,
                "authors": [],
                "published": None,
                "year": None,
                "source": "pdf",
                "domain": domain,
                "url": str(pdf_file),
                "metadata": {"file_name": pdf_file.name}
            })
        except Exception as e:
            print(f"PDF load error for {pdf_file.name}: {e}")

    return docs