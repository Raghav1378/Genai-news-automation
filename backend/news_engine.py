"""
News Intelligence Engine — v2
─────────────────────────────
1.  Tavily Search     → pull real-time news articles
2.  Groq LLM          → refine / analyse / summarise
3.  Sentiment & Bias  → extracted from LLM structured output
4.  Entity Extraction → people, orgs, locations
5.  Confidence Score  → computed from source quality + LLM certainty
6.  Recursive Research→ deep-dive on conflicting information
"""

import os
import json
import re
from dotenv import load_dotenv
from tavily import TavilyClient
from groq import Groq

# Load .env from root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYSTEM PROMPT — instructs the LLM to produce structured metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS_SYSTEM_PROMPT = """You are an elite news intelligence analyst. Given a user query and a set of real-time news articles retrieved from the web, you must:

1. **Synthesize** the information into a clear, well-structured analysis.
2. **Cite sources** by referencing the article titles / URLs where applicable.
3. **Identify key insights**, trends, and any contradictions across sources.

Write in a professional, concise tone. Use markdown formatting for readability (headers, bullet points, bold). Do NOT invent facts — only work with the provided sources.

At the VERY END of your analysis (after all content), output a JSON metadata block on a single line in this exact format:
<!--META:{"confidence":85,"sentiment":"positive","sentiment_score":72,"bias":"center","bias_score":15,"entities":[{"name":"Example Corp","type":"org","role":"Subject of analysis"},{"name":"John Doe","type":"person","role":"CEO mentioned"}],"contradictions_found":false}-->

Rules for the META block:
- confidence: 0-100 (how well-sourced is the answer)
- sentiment: "positive", "negative", "neutral", or "mixed"
- sentiment_score: 0-100 (intensity, 50 = perfectly neutral)
- bias: "left", "right", "center", "corporate", or "mixed"
- bias_score: 0-100 (intensity, 0 = no detectable bias)
- entities: array of key people, organizations, locations with type and role
- contradictions_found: true if sources significantly disagree"""


def fetch_news(query: str, max_results: int = 8, exclude_domains: list = None) -> dict:
    """Call Tavily to retrieve real-time news articles."""
    try:
        kwargs = dict(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
        )
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains
        response = tavily_client.search(**kwargs)
        return response
    except Exception as e:
        return {"results": [], "answer": None, "error": str(e)}


def _build_context(tavily_response: dict) -> str:
    """Format Tavily results into a context block for the LLM."""
    parts = []
    results = tavily_response.get("results", [])
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = r.get("content", "")
        score = r.get("score", 0)
        parts.append(
            f"### Source {i}: {title}\n"
            f"**URL:** {url}\n"
            f"**Relevance:** {score:.2f}\n"
            f"**Content:** {content}\n"
        )
    if tavily_response.get("answer"):
        parts.insert(0, f"**Tavily Quick Answer:** {tavily_response['answer']}\n\n---\n")
    return "\n---\n".join(parts)


def _extract_meta(llm_text: str) -> dict:
    """Pull the <!--META:{...}--> JSON block from the LLM output."""
    default = {
        "confidence": 50,
        "sentiment": "neutral",
        "sentiment_score": 50,
        "bias": "center",
        "bias_score": 10,
        "entities": [],
        "contradictions_found": False,
    }
    match = re.search(r"<!--META:(.*?)-->", llm_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            for key in default:
                if key not in parsed:
                    parsed[key] = default[key]
            return parsed
        except json.JSONDecodeError:
            pass
    return default


def _clean_analysis(llm_text: str) -> str:
    """Remove the META block from the visible analysis text."""
    cleaned = re.sub(r"\s*<!--META:.*?-->\s*", "", llm_text, flags=re.DOTALL).strip()
    return cleaned


def _compute_source_confidence(tavily_response: dict) -> float:
    """Average the Tavily relevance scores (0-1) → scale to 0-100."""
    results = tavily_response.get("results", [])
    if not results:
        return 0.0
    avg = sum(r.get("score", 0) for r in results) / len(results)
    return round(avg * 100, 1)


def refine_with_groq(query: str, context: str) -> str:
    """Send query + context to Groq LLM for deep analysis."""
    try:
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"## User Query\n{query}\n\n"
                        f"## Retrieved News Sources\n{context}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"**Error from Groq:** {str(e)}\n\n<!--META:{{\"confidence\":0,\"sentiment\":\"neutral\",\"sentiment_score\":50,\"bias\":\"center\",\"bias_score\":0,\"entities\":[],\"contradictions_found\":false}}-->"


def analyse(query: str, exclude_domains: list = None, deep_research: bool = False) -> dict:
    """
    Full pipeline:
      Tavily search → Groq refinement → structured metadata extraction.
    """
    # Step 1 — fetch real-time news
    tavily_resp = fetch_news(query, exclude_domains=exclude_domains)
    if tavily_resp.get("error"):
        return {
            "query": query,
            "analysis": f"⚠️ Error fetching news: {tavily_resp['error']}",
            "sources": [],
            "confidence": 0,
            "sentiment": "neutral",
            "sentiment_score": 50,
            "bias": "center",
            "bias_score": 0,
            "entities": [],
            "deep_research": False,
        }

    # Step 2 — build context & refine with LLM
    context = _build_context(tavily_resp)
    llm_analysis = refine_with_groq(query, context)

    # Step 3 — extract structured metadata
    meta = _extract_meta(llm_analysis)
    clean_text = _clean_analysis(llm_analysis)

    # Step 4 — blended confidence (60% LLM + 40% source quality)
    src_conf = _compute_source_confidence(tavily_resp)
    final_confidence = round(0.6 * meta["confidence"] + 0.4 * src_conf, 1)

    # Step 5 — extract sources
    sources = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "score": round(r.get("score", 0), 3),
        }
        for r in tavily_resp.get("results", [])
    ]

    # Step 6 — recursive deep research if requested and contradictions found
    deep_used = False
    if deep_research and meta.get("contradictions_found"):
        sub_queries = _generate_sub_queries(query, clean_text)
        for sq in sub_queries[:2]:
            sub_resp = fetch_news(sq, max_results=4, exclude_domains=exclude_domains)
            if not sub_resp.get("error"):
                sub_context = _build_context(sub_resp)
                sub_analysis = refine_with_groq(
                    f"Verify and clarify: {sq}",
                    sub_context
                )
                sub_clean = _clean_analysis(sub_analysis)
                clean_text += f"\n\n---\n\n### 🔍 Deep Research: {sq}\n\n{sub_clean}"
                sources.extend([
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "score": round(r.get("score", 0), 3),
                    }
                    for r in sub_resp.get("results", [])
                ])
        deep_used = True

    return {
        "query": query,
        "analysis": clean_text,
        "sources": sources,
        "confidence": final_confidence,
        "sentiment": meta["sentiment"],
        "sentiment_score": meta["sentiment_score"],
        "bias": meta["bias"],
        "bias_score": meta["bias_score"],
        "entities": meta["entities"],
        "deep_research": deep_used,
    }


def _generate_sub_queries(original_query: str, analysis: str) -> list:
    """Use Groq to generate 2 follow-up verification queries."""
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You generate follow-up search queries to verify contradictions in news. Return ONLY a JSON array of 2 query strings, nothing else.",
                },
                {
                    "role": "user",
                    "content": f"Original query: {original_query}\n\nAnalysis with contradictions:\n{analysis[:1500]}\n\nGenerate 2 verification queries:",
                },
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return [f"verify {original_query}", f"{original_query} fact check"]
