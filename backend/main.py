"""
FastAPI Backend for News Intelligence System — v3
──────────────────────────────────────────────────
Endpoints:
  POST /api/search              — run news query (Groq + Gemini + FactCheck)
  POST /api/search/deep         — recursive deep research mode
  GET  /api/history             — list all past searches
  GET  /api/history/{id}        — get a specific search
  DELETE /api/history/{id}      — delete one entry
  DELETE /api/history           — clear all history
  GET  /api/blacklist           — list blacklisted domains
  POST /api/blacklist           — add a domain to blacklist
  DELETE /api/blacklist/{id}    — remove a blacklisted domain
  GET  /api/watchlist           — list watched topics
  POST /api/watchlist           — add a topic to watch
  DELETE /api/watchlist/{id}    — remove a watched topic
  GET  /api/trends              — sentiment/confidence trends over time
  GET  /api/entities/network    — entity co-occurrence network
  GET  /api/export/pdf/{id}     — download PDF report
  POST /api/briefing/generate   — generate email digest from watchlist
  GET  /api/briefing/preview    — preview email digest HTML
  GET  /api/email-settings      — get SMTP email settings
  POST /api/email-settings      — save SMTP email settings
  GET  /api/languages           — list supported languages
  POST /api/memory/search       — semantic search over past analyses
  GET  /api/memory/stats        — vector store stats
  POST /api/voice/briefing/{id} — generate voice briefing script
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from database import (
    init_db,
    save_search, get_all_history, get_search_by_id, delete_search, clear_history,
    add_blacklisted_domain, get_blacklisted_domains, delete_blacklisted_domain,
    add_watchlist_topic, get_watchlists, delete_watchlist_topic, update_watchlist_checked,
    get_email_settings, save_email_settings,
)
from news_engine import analyse, _build_context, fetch_news
from gemini_engine import gemini_analyse
from trend_service import get_topic_trends, get_entity_network
from pdf_service import generate_report_pdf
from email_service import generate_briefing_html, send_briefing
from translate_engine import get_supported_languages, translate_query, translate_results_to_english
from fact_checker import fact_check_analysis, balance_bias

# Optional: Vector store (may fail if chromadb not installed)
try:
    from vector_store import index_search as vs_index, query_memory as vs_query, get_memory_stats as vs_stats
    VECTOR_AVAILABLE = True
except Exception:
    VECTOR_AVAILABLE = False


# ── Keep-Alive Self-Ping (prevents Render free tier spin-down) ────
import asyncio
import httpx

RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "")  # Render sets this automatically

async def keep_alive():
    """Ping our own /health endpoint every 13 min to prevent spin-down."""
    await asyncio.sleep(60)  # wait for server to fully start
    async with httpx.AsyncClient() as client:
        while True:
            try:
                url = RENDER_URL or "http://localhost:8000"
                await client.get(f"{url}/health", timeout=10)
                print("[KeepAlive] Ping OK")
            except Exception:
                pass
            await asyncio.sleep(780)  # 13 minutes


# ── Lifespan ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Start keep-alive background task
    task = asyncio.create_task(keep_alive())
    yield
    task.cancel()


app = FastAPI(title="News Intelligence API", version="3.0.0", lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {"status": "ok"}

# ── CORS ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Per-Request API Key Injection ─────────────────────────
@app.middleware("http")
async def inject_api_keys(request: Request, call_next):
    """
    Reads X-Groq-Key and X-Gemini-Key headers and overrides
    module-level API clients so each user uses their own keys.
    Falls back to .env keys if headers are not provided.
    """
    import news_engine
    import gemini_engine
    import fact_checker
    import translate_engine
    import vector_store
    from groq import Groq

    groq_key = request.headers.get("x-groq-key") or os.getenv("GROQ_API_KEY", "")
    gemini_key = request.headers.get("x-gemini-key") or os.getenv("GOOGLE_API_KEY", "")

    # Override Groq clients
    if groq_key:
        news_engine.groq_client = Groq(api_key=groq_key)
        fact_checker.groq_client = Groq(api_key=groq_key)

    # Override Gemini key for lazy-init modules
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key
        # Reset cached models so they re-init with new key
        gemini_engine._gemini_model = None
        translate_engine._get_gemini = translate_engine._get_gemini  # keep original
        vector_store._embed_model = None

    response = await call_next(request)
    return response


# ── Validate Keys Endpoint ────────────────────────────────
class ValidateKeysRequest(BaseModel):
    groq_key: str = ""
    gemini_key: str = ""


@app.post("/api/validate-keys")
async def validate_keys(req: ValidateKeysRequest):
    """
    Test if provided API keys are valid.
    At least one key (Groq or Gemini) must work.
    """
    results = {"groq": False, "gemini": False, "valid": False}

    # Test Groq key
    if req.groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=req.groq_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5,
            )
            if resp.choices:
                results["groq"] = True
        except Exception as e:
            results["groq_error"] = str(e)[:100]

    # Test Gemini key
    if req.gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=req.gemini_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content("Say OK")
            if resp.text:
                results["gemini"] = True
        except Exception as e:
            results["gemini_error"] = str(e)[:100]

    results["valid"] = results["groq"] or results["gemini"]
    return results


# ── Request / Response Models ─────────────────────────────
class SearchRequest(BaseModel):
    query: str
    max_results: int = 8
    deep_research: bool = False
    language: str = "en"


class EntityItem(BaseModel):
    name: str
    type: str
    role: str


class SourceItem(BaseModel):
    title: str
    url: str
    score: float


class FactCheckItem(BaseModel):
    claim: str
    status: str
    evidence: str


class SearchResponse(BaseModel):
    id: int
    query: str
    analysis: str
    sources: list[SourceItem]
    confidence: float
    sentiment: str
    sentiment_score: float
    bias: str
    bias_score: float
    entities: list
    gemini_analysis: str
    gemini_confidence: float
    deep_research: bool
    fact_check_results: list
    bias_balance: str
    created_at: str


class DomainRequest(BaseModel):
    domain: str


class TopicRequest(BaseModel):
    topic: str


class MemoryRequest(BaseModel):
    query: str
    top_k: int = 5


class EmailSettingsRequest(BaseModel):
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    email: str = ""
    app_password: str = ""
    recipient: str = ""
    is_active: bool = False


# ── Search Routes ─────────────────────────────────────────
@app.post("/api/search", response_model=SearchResponse)
async def search_news(req: SearchRequest):
    """Run full Tavily → Groq → Gemini → FactCheck pipeline, persist, and return."""
    # Get blacklisted domains
    bl = await get_blacklisted_domains()
    exclude = [d["domain"] for d in bl] if bl else None

    # Handle cross-lingual search
    search_query = req.query
    if req.language != "en":
        search_query = translate_query(req.query, req.language)

    # Run Groq analysis
    result = analyse(search_query, exclude_domains=exclude, deep_research=req.deep_research)

    # If cross-lingual, translate sources back
    if req.language != "en" and result.get("sources"):
        result["sources"] = translate_results_to_english(result["sources"], req.language)

    # Run Gemini analysis (multi-model voting)
    tavily_resp = fetch_news(search_query, exclude_domains=exclude)
    context = _build_context(tavily_resp) if not tavily_resp.get("error") else ""
    gemini_result = gemini_analyse(req.query, context, groq_summary=result["analysis"][:1500])

    # Run fact-check
    fact_checks = fact_check_analysis(result["analysis"], result["sources"])

    # Run bias balance
    bias_bal = balance_bias(req.query, result["analysis"], result.get("bias", "center"))
    bias_balance_text = bias_bal.get("perspective", "")

    # Persist
    row_id = await save_search(
        query=req.query,  # Always store original query
        result_summary=result["analysis"],
        sources=result["sources"],
        confidence=result["confidence"],
        sentiment=result["sentiment"],
        sentiment_score=result["sentiment_score"],
        bias=result["bias"],
        bias_score=result["bias_score"],
        entities=result["entities"],
        gemini_analysis=gemini_result["analysis"],
        gemini_confidence=gemini_result["confidence"],
        deep_research=result["deep_research"],
        fact_check_results=fact_checks,
        bias_balance=bias_balance_text,
    )

    # Index into vector store (async-safe, fire and forget)
    if VECTOR_AVAILABLE:
        try:
            vs_index(
                row_id,
                req.query,
                result["analysis"],
                {"confidence": result["confidence"], "sentiment": result["sentiment"]},
            )
        except Exception:
            pass

    saved = await get_search_by_id(row_id)
    return _to_response(saved)


@app.post("/api/search/deep", response_model=SearchResponse)
async def deep_search_news(req: SearchRequest):
    """Explicit deep research endpoint — forces recursive research."""
    req.deep_research = True
    return await search_news(req)


# ── History Routes ────────────────────────────────────────
@app.get("/api/history")
async def list_history():
    return await get_all_history()


@app.get("/api/history/{search_id}")
async def get_history_item(search_id: int):
    item = await get_search_by_id(search_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Not found")
    return item


@app.delete("/api/history/{search_id}")
async def delete_history_item(search_id: int):
    await delete_search(search_id)
    return {"status": "deleted", "id": search_id}


@app.delete("/api/history")
async def clear_all_history():
    await clear_history()
    return {"status": "cleared"}


# ── Blacklist Routes ──────────────────────────────────────
@app.get("/api/blacklist")
async def list_blacklist():
    return await get_blacklisted_domains()


@app.post("/api/blacklist")
async def add_to_blacklist(req: DomainRequest):
    domain_id = await add_blacklisted_domain(req.domain)
    return {"status": "added", "id": domain_id, "domain": req.domain}


@app.delete("/api/blacklist/{domain_id}")
async def remove_from_blacklist(domain_id: int):
    await delete_blacklisted_domain(domain_id)
    return {"status": "deleted", "id": domain_id}


# ── Watchlist Routes ──────────────────────────────────────
@app.get("/api/watchlist")
async def list_watchlist():
    return await get_watchlists()


@app.post("/api/watchlist")
async def add_to_watchlist(req: TopicRequest):
    topic_id = await add_watchlist_topic(req.topic)
    return {"status": "added", "id": topic_id, "topic": req.topic}


@app.delete("/api/watchlist/{topic_id}")
async def remove_from_watchlist(topic_id: int):
    await delete_watchlist_topic(topic_id)
    return {"status": "deleted", "id": topic_id}


# ── Trend Routes ──────────────────────────────────────────
@app.get("/api/trends")
async def get_trends(topic: str = None, days: int = 30):
    """Return sentiment/confidence trend data from search history."""
    return await get_topic_trends(topic, days)


@app.get("/api/entities/network")
async def get_entity_map(search_id: int = None):
    """Return entity co-occurrence network."""
    return await get_entity_network(search_id)


# ── PDF Export ────────────────────────────────────────────
@app.get("/api/export/pdf/{search_id}")
async def export_pdf(search_id: int):
    """Generate and download a professional PDF report."""
    item = await get_search_by_id(search_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Search not found")

    pdf_bytes = generate_report_pdf(item)
    filename = f"NeuralPulse_Report_{search_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Email Briefing ────────────────────────────────────────
@app.post("/api/briefing/generate")
async def generate_email_briefing(send_email: bool = False):
    """Generate a briefing from all active watchlist topics. Optionally send via email."""
    watchlists = await get_watchlists()
    active_topics = [w for w in watchlists if w.get("is_active")]

    if not active_topics:
        raise HTTPException(status_code=400, detail="No active watchlist topics. Add topics first.")

    # Run search for each topic
    topics_results = []
    for w in active_topics[:5]:  # Cap at 5 to avoid rate limits
        result = analyse(w["topic"])
        topics_results.append({
            "topic": w["topic"],
            "analysis": result["analysis"],
            "confidence": result["confidence"],
            "sentiment": result["sentiment"],
            "sources": result["sources"],
        })
        await update_watchlist_checked(w["id"])

    html = generate_briefing_html(topics_results)

    response = {"status": "generated", "html": html, "topics_count": len(topics_results)}

    if send_email:
        settings = await get_email_settings()
        if not settings.get("email") or not settings.get("app_password"):
            response["email_status"] = "SMTP not configured"
        else:
            email_result = send_briefing(
                to_email=settings["recipient"] or settings["email"],
                html_content=html,
                smtp_host=settings["smtp_host"],
                smtp_port=settings["smtp_port"],
                sender_email=settings["email"],
                sender_password=settings["app_password"],
            )
            response["email_status"] = email_result

    return response


@app.get("/api/briefing/preview")
async def preview_briefing():
    """Preview the email briefing HTML without sending."""
    watchlists = await get_watchlists()
    active_topics = [w for w in watchlists if w.get("is_active")]

    if not active_topics:
        return {"html": "<p>No active watchlist topics.</p>"}

    # Use last known results from history or simple summaries
    topics_results = []
    for w in active_topics[:5]:
        topics_results.append({
            "topic": w["topic"],
            "analysis": f"Briefing preview for: {w['topic']}. Run full generation to get real analysis.",
            "confidence": 0,
            "sentiment": "neutral",
            "sources": [],
        })

    html = generate_briefing_html(topics_results)
    return {"html": html}


# ── Email Settings ────────────────────────────────────────
@app.get("/api/email-settings")
async def get_email_cfg():
    settings = await get_email_settings()
    # Mask password for security
    if settings.get("app_password"):
        settings["app_password"] = "••••••••"
    return settings


@app.post("/api/email-settings")
async def save_email_cfg(req: EmailSettingsRequest):
    # If password is masked, don't overwrite
    current = await get_email_settings()
    password = req.app_password if req.app_password != "••••••••" else current.get("app_password", "")
    await save_email_settings(
        smtp_host=req.smtp_host,
        smtp_port=req.smtp_port,
        email=req.email,
        app_password=password,
        recipient=req.recipient,
        is_active=req.is_active,
    )
    return {"status": "saved"}


# ── Languages ─────────────────────────────────────────────
@app.get("/api/languages")
async def list_languages():
    return get_supported_languages()


# ── Vector Memory ─────────────────────────────────────────
@app.post("/api/memory/search")
async def search_memory(req: MemoryRequest):
    if not VECTOR_AVAILABLE:
        return {"results": [], "error": "Vector store not available. Install chromadb."}
    results = vs_query(req.query, req.top_k)
    return {"results": results}


@app.get("/api/memory/stats")
async def memory_stats():
    if not VECTOR_AVAILABLE:
        return {"count": 0, "status": "unavailable"}
    return vs_stats()


# ── Voice Briefing ────────────────────────────────────────
@app.post("/api/voice/briefing/{search_id}")
async def generate_voice_briefing(search_id: int):
    """Generate a concise 2-minute voice script from an analysis."""
    item = await get_search_by_id(search_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Search not found")

    # Use Gemini to generate a concise spoken script
    from gemini_engine import _get_model
    model = _get_model()
    if model is None:
        return {"script": item.get("result_summary", "")[:500], "error": "Gemini unavailable"}

    try:
        prompt = (
            "Convert the following news analysis into a concise, engaging 2-minute spoken briefing. "
            "Write it as if you're a news anchor delivering a brief. Use clear, conversational language. "
            "Do NOT use markdown. Keep it under 300 words.\n\n"
            f"Query: {item['query']}\n\n"
            f"Analysis:\n{item['result_summary'][:3000]}"
        )
        response = model.generate_content(prompt)
        return {"script": response.text.strip(), "query": item["query"]}
    except Exception as e:
        return {"script": item.get("result_summary", "")[:500], "error": str(e)}


# ── Helper ────────────────────────────────────────────────
def _to_response(saved: dict) -> SearchResponse:
    return SearchResponse(
        id=saved["id"],
        query=saved["query"],
        analysis=saved["result_summary"],
        sources=saved["sources"],
        confidence=saved["confidence"],
        sentiment=saved.get("sentiment", "neutral"),
        sentiment_score=saved.get("sentiment_score", 50),
        bias=saved.get("bias", "center"),
        bias_score=saved.get("bias_score", 0),
        entities=saved.get("entities", []),
        gemini_analysis=saved.get("gemini_analysis", ""),
        gemini_confidence=saved.get("gemini_confidence", 0),
        deep_research=saved.get("deep_research", False),
        fact_check_results=saved.get("fact_check_results", []),
        bias_balance=saved.get("bias_balance", ""),
        created_at=saved["created_at"],
    )


# ── Serve Frontend ────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
