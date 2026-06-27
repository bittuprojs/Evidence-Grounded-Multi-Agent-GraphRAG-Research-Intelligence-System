from typing import List, Dict


class CitationManager:
    def format_sources(self, sources: List[Dict]) -> str:
        """
        Format sources into a clean bibliography-style block.
        """
        if not sources:
            return "No sources available."

        lines = []
        for i, s in enumerate(sources, 1):
            title = s.get("title", "Unknown title")
            year = s.get("year", "n.d.")
            source = s.get("source", "unknown")
            url = s.get("url", "")
            lines.append(f"[{i}] {title} ({year}) - {source}")
            if url:
                lines.append(f"    {url}")

        return "\n".join(lines)

    def format_inline_citations(self, sources: List[Dict]) -> str:
        """
        Return a compact inline citation string.
        """
        if not sources:
            return ""

        parts = []
        for i, s in enumerate(sources, 1):
            title = s.get("title", "Unknown")
            year = s.get("year", "n.d.")
            parts.append(f"{title} ({year})")

        return "; ".join(parts)

    def export_bibliography(self, sources: List[Dict]) -> str:
        """
        Create a plain-text bibliography output.
        """
        return self.format_sources(sources)