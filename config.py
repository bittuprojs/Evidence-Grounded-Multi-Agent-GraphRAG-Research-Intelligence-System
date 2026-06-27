from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 120
DEFAULT_TOP_K = 5

DOMAINS = ["AI/ML", "Medical", "Finance", "Science", "General"]