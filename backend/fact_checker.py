"""
Fact Checker — Claim Verification & Bias Balancer
───────────────────────────────────────────────────
Cross-references claims from the analysis and detects bias imbalance.
Uses Groq LLM with specialized prompts.
"""

import os
import json
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

FACT_CHECK_PROMPT = """You are a rigorous fact-checking analyst. Given a news analysis and its sources, extract the key factual claims and assess each one.

For EACH claim, provide:
- "claim": the factual statement (one sentence)
- "status": one of "verified" (multiple sources agree), "unconfirmed" (only one source, can't verify), or "contradictory" (sources disagree)
- "evidence": brief explanation of why you assigned that status

Return ONLY a JSON array of claim objects. Example:
[
  {"claim": "Company X acquired Y for $1B", "status": "verified", "evidence": "Confirmed by 3 independent sources"},
  {"claim": "The deal will close in Q2", "status": "unconfirmed", "evidence": "Only mentioned by one anonymous source"}
]

Maximum 8 claims. Focus on the most important factual statements."""


BIAS_BALANCE_PROMPT = """You are a media balance analyst. The current news coverage on this topic shows a "{bias}" bias.

Given the original query and the biased analysis, write a SHORT counter-perspective (3-4 sentences) that presents the opposing viewpoint fairly. Be factual and measured — do not be inflammatory.

Return JSON: {{"perspective": "your counter-perspective text", "balance_note": "brief note on what bias was detected"}}"""


def fact_check_analysis(analysis: str, sources: list[dict] = None) -> list[dict]:
    """
    Extract and verify claims from the analysis text.
    Returns list of {claim, status, evidence}.
    """
    source_info = ""
    if sources:
        for i, src in enumerate(sources[:6], 1):
            source_info += f"\nSource {i}: {src.get('title', '')} ({src.get('url', '')})"

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": FACT_CHECK_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"## Analysis to fact-check:\n{analysis[:3000]}\n\n"
                        f"## Available sources:{source_info}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=1500,
        )

        raw = resp.choices[0].message.content.strip()
        # Extract JSON array
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            claims = json.loads(json_match.group())
            # Validate structure
            valid = []
            for c in claims:
                if isinstance(c, dict) and "claim" in c:
                    valid.append({
                        "claim": c.get("claim", ""),
                        "status": c.get("status", "unconfirmed"),
                        "evidence": c.get("evidence", ""),
                    })
            return valid[:8]
        return []

    except Exception as e:
        print(f"[FactChecker] Error: {e}")
        return []


def balance_bias(query: str, analysis: str, current_bias: str) -> dict:
    """
    If bias is detected, generate a counter-perspective to balance coverage.
    Returns {perspective, balance_note} or empty dict if balanced.
    """
    if current_bias in ("center", "neutral", ""):
        return {"perspective": "", "balance_note": "Coverage appears balanced."}

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": BIAS_BALANCE_PROMPT.format(bias=current_bias)},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"Current analysis (shows {current_bias} bias):\n{analysis[:2000]}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=500,
        )

        raw = resp.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "perspective": result.get("perspective", ""),
                "balance_note": result.get("balance_note", ""),
            }
        return {"perspective": raw, "balance_note": f"Detected {current_bias} bias"}

    except Exception as e:
        print(f"[BiasBalancer] Error: {e}")
        return {"perspective": "", "balance_note": f"Error: {str(e)}"}
