"""
Translate Engine — Cross-Lingual News Intelligence
────────────────────────────────────────────────────
Uses Gemini to translate queries and results for global news coverage.
Supports searching non-English sources and translating back to English.
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

SUPPORTED_LANGUAGES = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "ar": "Arabic",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
}


def _get_gemini():
    """Lazy-init Gemini model."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        print(f"[Translate] Gemini init failed: {e}")
        return None


def translate_query(query: str, target_lang: str) -> str:
    """Translate a search query from English to target language."""
    if target_lang == "en":
        return query

    lang_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
    model = _get_gemini()
    if not model:
        return query

    try:
        resp = model.generate_content(
            f"Translate the following search query to {lang_name}. "
            f"Return ONLY the translated text, nothing else.\n\n"
            f"Query: {query}"
        )
        return resp.text.strip()
    except Exception:
        return query


def translate_results_to_english(results: list[dict], source_lang: str) -> list[dict]:
    """Translate article titles and content back to English."""
    if source_lang == "en" or not results:
        return results

    lang_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
    model = _get_gemini()
    if not model:
        return results

    # Batch translate titles and content summaries
    items = []
    for r in results[:8]:
        items.append({
            "title": r.get("title", ""),
            "content": r.get("content", "")[:300],
        })

    try:
        prompt = (
            f"Translate the following news article data from {lang_name} to English. "
            f"Return a JSON array with the same structure.\n\n"
            f"```json\n{json.dumps(items, ensure_ascii=False)}\n```"
        )
        resp = model.generate_content(prompt)
        raw = resp.text.strip()

        # Extract JSON from response
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            translated = json.loads(json_match.group())
            for i, t in enumerate(translated):
                if i < len(results):
                    results[i]["title"] = t.get("title", results[i].get("title", ""))
                    results[i]["content"] = t.get("content", results[i].get("content", ""))
                    results[i]["original_language"] = lang_name
    except Exception as e:
        print(f"[Translate] Translation failed: {e}")

    return results


def get_supported_languages() -> dict:
    """Return the map of supported language codes to names."""
    return SUPPORTED_LANGUAGES
