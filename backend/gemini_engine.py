"""
Gemini Engine — Multi-Model Voting
───────────────────────────────────
Sends the same news context to Google Gemini for an independent analysis.
Used to cross-verify the Groq output and highlight agreement / disagreement.
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

_gemini_model = None


def _get_model():
    """Lazy-init the Gemini model to avoid import errors if not installed."""
    global _gemini_model
    if _gemini_model is None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            _gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        except Exception as e:
            print(f"[Gemini] Init failed: {e}")
            return None
    return _gemini_model


GEMINI_SYSTEM = """You are a senior news intelligence analyst providing a second opinion. Given a user query and retrieved news sources, provide:

1. A concise, independent analysis (use markdown).
2. At the END, output a single-line JSON metadata block:
<!--META:{"confidence":75,"agreement_notes":"Brief note on what you agree/disagree with compared to a first-pass analysis"}-->

Be factual, professional, and concise. Do NOT invent information."""


def gemini_analyse(query: str, context: str, groq_summary: str = "") -> dict:
    """
    Run Gemini analysis on the same context.
    Returns dict with analysis text, confidence, and agreement notes.
    """
    model = _get_model()
    if model is None:
        return {
            "analysis": "⚠️ Gemini model unavailable — skipping multi-model verification.",
            "confidence": 0,
            "agreement_notes": "Model not available",
            "available": False,
        }

    prompt = (
        f"{GEMINI_SYSTEM}\n\n"
        f"## User Query\n{query}\n\n"
        f"## Retrieved News Sources\n{context}\n\n"
    )
    if groq_summary:
        prompt += f"## First-Pass Analysis (from another model)\n{groq_summary[:2000]}\n\n"
    prompt += "Provide your independent analysis now:"

    try:
        response = model.generate_content(prompt)
        raw = response.text

        # Extract meta
        meta = {"confidence": 50, "agreement_notes": ""}
        match = re.search(r"<!--META:(.*?)-->", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                meta.update(parsed)
            except json.JSONDecodeError:
                pass

        # Clean the analysis text
        clean = re.sub(r"\s*<!--META:.*?-->\s*", "", raw, flags=re.DOTALL).strip()

        return {
            "analysis": clean,
            "confidence": min(meta.get("confidence", 50), 100),
            "agreement_notes": meta.get("agreement_notes", ""),
            "available": True,
        }

    except Exception as e:
        return {
            "analysis": f"⚠️ Gemini error: {str(e)}",
            "confidence": 0,
            "agreement_notes": "Error occurred",
            "available": False,
        }
