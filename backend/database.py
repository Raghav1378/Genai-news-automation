"""
Database module — SQLite async persistence layer (v2)
─────────────────────────────────────────────────────
Tables:
  search_history    — stores queries, results, sentiment, entities, and model comparisons
  blacklisted_domains — domains to exclude from future searches
  watchlists        — topics the user is monitoring
"""

import aiosqlite
import json
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "news_intelligence.db")


async def init_db():
    """Create / migrate all tables."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Main search history table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                query              TEXT    NOT NULL,
                result_summary     TEXT,
                sources            TEXT,
                confidence         REAL,
                sentiment          TEXT    DEFAULT 'neutral',
                sentiment_score    REAL    DEFAULT 50,
                bias               TEXT    DEFAULT 'center',
                bias_score         REAL    DEFAULT 0,
                entities           TEXT    DEFAULT '[]',
                gemini_analysis    TEXT    DEFAULT '',
                gemini_confidence  REAL    DEFAULT 0,
                deep_research      INTEGER DEFAULT 0,
                created_at         TEXT    NOT NULL
            )
        """)

        # Blacklisted domains
        await db.execute("""
            CREATE TABLE IF NOT EXISTS blacklisted_domains (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                domain     TEXT    UNIQUE NOT NULL,
                created_at TEXT    NOT NULL
            )
        """)

        # Watchlists
        await db.execute("""
            CREATE TABLE IF NOT EXISTS watchlists (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                topic        TEXT    NOT NULL,
                is_active    INTEGER DEFAULT 1,
                last_checked TEXT,
                created_at   TEXT    NOT NULL
            )
        """)

        # Email settings
        await db.execute("""
            CREATE TABLE IF NOT EXISTS email_settings (
                id           INTEGER PRIMARY KEY CHECK (id = 1),
                smtp_host    TEXT    DEFAULT 'smtp.gmail.com',
                smtp_port    INTEGER DEFAULT 587,
                email        TEXT    DEFAULT '',
                app_password TEXT    DEFAULT '',
                recipient    TEXT    DEFAULT '',
                is_active    INTEGER DEFAULT 0
            )
        """)
        # Ensure a default row exists
        await db.execute("""
            INSERT OR IGNORE INTO email_settings (id) VALUES (1)
        """)

        # Migrate: add new columns to existing table if they don't exist
        try:
            await db.execute("ALTER TABLE search_history ADD COLUMN sentiment TEXT DEFAULT 'neutral'")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE search_history ADD COLUMN sentiment_score REAL DEFAULT 50")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE search_history ADD COLUMN bias TEXT DEFAULT 'center'")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE search_history ADD COLUMN bias_score REAL DEFAULT 0")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE search_history ADD COLUMN entities TEXT DEFAULT '[]'")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE search_history ADD COLUMN gemini_analysis TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE search_history ADD COLUMN gemini_confidence REAL DEFAULT 0")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE search_history ADD COLUMN deep_research INTEGER DEFAULT 0")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE search_history ADD COLUMN fact_check_results TEXT DEFAULT '[]'")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE search_history ADD COLUMN bias_balance TEXT DEFAULT ''")
        except Exception:
            pass

        await db.commit()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEARCH HISTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_row(row) -> dict:
    """Convert a Row to a dict with JSON-parsed fields."""
    return {
        "id": row["id"],
        "query": row["query"],
        "result_summary": row["result_summary"],
        "sources": json.loads(row["sources"]) if row["sources"] else [],
        "confidence": row["confidence"] or 0,
        "sentiment": row["sentiment"] or "neutral",
        "sentiment_score": row["sentiment_score"] if row["sentiment_score"] is not None else 50,
        "bias": row["bias"] or "center",
        "bias_score": row["bias_score"] if row["bias_score"] is not None else 0,
        "entities": json.loads(row["entities"]) if row["entities"] else [],
        "gemini_analysis": row["gemini_analysis"] or "",
        "gemini_confidence": row["gemini_confidence"] if row["gemini_confidence"] is not None else 0,
        "deep_research": bool(row["deep_research"]) if row["deep_research"] is not None else False,
        "fact_check_results": json.loads(row["fact_check_results"]) if row["fact_check_results"] else [],
        "bias_balance": row["bias_balance"] or "",
        "created_at": row["created_at"],
    }


async def save_search(
    query: str,
    result_summary: str,
    sources: list,
    confidence: float,
    sentiment: str = "neutral",
    sentiment_score: float = 50,
    bias: str = "center",
    bias_score: float = 0,
    entities: list = None,
    gemini_analysis: str = "",
    gemini_confidence: float = 0,
    deep_research: bool = False,
    fact_check_results: list = None,
    bias_balance: str = "",
) -> int:
    """Persist a search and return its row id."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO search_history
                (query, result_summary, sources, confidence, sentiment, sentiment_score,
                 bias, bias_score, entities, gemini_analysis, gemini_confidence,
                 deep_research, fact_check_results, bias_balance, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query,
                result_summary,
                json.dumps(sources),
                confidence,
                sentiment,
                sentiment_score,
                bias,
                bias_score,
                json.dumps(entities or []),
                gemini_analysis,
                gemini_confidence,
                1 if deep_research else 0,
                json.dumps(fact_check_results or []),
                bias_balance,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await db.commit()
        return cursor.lastrowid


async def get_all_history():
    """Return all past searches, newest first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM search_history ORDER BY id DESC"
        )
        rows = await cursor.fetchall()
        return [_parse_row(row) for row in rows]


async def get_search_by_id(search_id: int):
    """Return a single search by its id."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM search_history WHERE id = ?", (search_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _parse_row(row)


async def delete_search(search_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM search_history WHERE id = ?", (search_id,))
        await db.commit()


async def clear_history():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM search_history")
        await db.commit()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BLACKLISTED DOMAINS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def add_blacklisted_domain(domain: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT OR IGNORE INTO blacklisted_domains (domain, created_at) VALUES (?, ?)",
            (domain.lower().strip(), datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()
        return cursor.lastrowid


async def get_blacklisted_domains() -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM blacklisted_domains ORDER BY id DESC")
        rows = await cursor.fetchall()
        return [{"id": row["id"], "domain": row["domain"], "created_at": row["created_at"]} for row in rows]


async def delete_blacklisted_domain(domain_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM blacklisted_domains WHERE id = ?", (domain_id,))
        await db.commit()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WATCHLISTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def add_watchlist_topic(topic: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO watchlists (topic, created_at) VALUES (?, ?)",
            (topic.strip(), datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()
        return cursor.lastrowid


async def get_watchlists() -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM watchlists ORDER BY id DESC")
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "topic": row["topic"],
                "is_active": bool(row["is_active"]),
                "last_checked": row["last_checked"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]


async def delete_watchlist_topic(topic_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM watchlists WHERE id = ?", (topic_id,))
        await db.commit()


async def update_watchlist_checked(topic_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE watchlists SET last_checked = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), topic_id),
        )
        await db.commit()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMAIL SETTINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def get_email_settings() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM email_settings WHERE id = 1")
        row = await cursor.fetchone()
        if row is None:
            return {
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "email": "",
                "app_password": "",
                "recipient": "",
                "is_active": False,
            }
        return {
            "smtp_host": row["smtp_host"] or "smtp.gmail.com",
            "smtp_port": row["smtp_port"] or 587,
            "email": row["email"] or "",
            "app_password": row["app_password"] or "",
            "recipient": row["recipient"] or "",
            "is_active": bool(row["is_active"]),
        }


async def save_email_settings(
    smtp_host: str, smtp_port: int, email: str,
    app_password: str, recipient: str, is_active: bool
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE email_settings
               SET smtp_host = ?, smtp_port = ?, email = ?,
                   app_password = ?, recipient = ?, is_active = ?
               WHERE id = 1""",
            (smtp_host, smtp_port, email, app_password, recipient, 1 if is_active else 0),
        )
        await db.commit()

