import os
from typing import Optional

try:
    from google import genai
    from google.genai import types
except Exception as e:
    genai = None
    types = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class GeminiLLM:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        if genai is None:
            raise RuntimeError(
                f"Gemini SDK import failed: {_IMPORT_ERROR}. "
                "Install google-genai in the same Python environment as Streamlit."
            )

        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return (getattr(response, "text", None) or str(response)).strip()
        except Exception as e:
            return f"LLM generation failed: {e}"
        