import arxiv
from datetime import datetime
from typing import List, Dict, Optional

def fetch_arxiv_papers(
    query: str,
    max_results: int = 10,
    domain: str = "General"
) -> List[Dict]:
    """
    Fetch paper metadata from ArXiv.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []

    try:
        for result in client.results(search):
            papers.append({
                "doc_id": result.entry_id.split("/")[-1],
                "title": result.title.strip(),
                "text": result.summary.strip(),
                "authors": [str(a) for a in result.authors],
                "published": result.published.strftime("%Y-%m-%d") if result.published else None,
                "year": result.published.year if result.published else None,
                "source": "arxiv",
                "domain": domain,
                "url": result.pdf_url,
                "metadata": {
                    "entry_id": result.entry_id,
                    "categories": result.categories,
                    "query": query,
                    "collected_at": datetime.now().isoformat()
                }
            })
    except Exception as e:
        print(f"ArXiv fetch error for query '{query}': {e}")

    return papers