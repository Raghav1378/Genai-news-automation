"""
Trend Service — Historical Sentiment & Entity Network Analysis
──────────────────────────────────────────────────────────────
Queries search_history to compute time-series trends and entity co-occurrence.
"""

import aiosqlite
import json
from collections import defaultdict
from database import DB_PATH


async def get_topic_trends(topic: str = None, days: int = 30) -> dict:
    """
    Return time-series sentiment/confidence data from past searches.
    If topic is provided, filters history to matching queries.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if topic:
            cursor = await db.execute(
                """SELECT query, confidence, sentiment, sentiment_score, bias_score, created_at
                   FROM search_history
                   WHERE query LIKE ? AND created_at >= datetime('now', ?)
                   ORDER BY created_at ASC""",
                (f"%{topic}%", f"-{days} days"),
            )
        else:
            cursor = await db.execute(
                """SELECT query, confidence, sentiment, sentiment_score, bias_score, created_at
                   FROM search_history
                   WHERE created_at >= datetime('now', ?)
                   ORDER BY created_at ASC""",
                (f"-{days} days",),
            )
        rows = await cursor.fetchall()

    dates = []
    confidences = []
    sentiment_scores = []
    bias_scores = []
    queries = []

    for row in rows:
        dates.append(row["created_at"])
        confidences.append(row["confidence"] or 0)
        sentiment_scores.append(row["sentiment_score"] if row["sentiment_score"] is not None else 50)
        bias_scores.append(row["bias_score"] if row["bias_score"] is not None else 0)
        queries.append(row["query"])

    return {
        "dates": dates,
        "confidences": confidences,
        "sentiment_scores": sentiment_scores,
        "bias_scores": bias_scores,
        "queries": queries,
        "count": len(dates),
    }


async def get_entity_network(search_id: int = None) -> dict:
    """
    Build an entity co-occurrence network from search history.
    If search_id is given, only for that search. Otherwise, last 20 searches.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if search_id:
            cursor = await db.execute(
                "SELECT entities FROM search_history WHERE id = ?", (search_id,)
            )
        else:
            cursor = await db.execute(
                "SELECT entities FROM search_history ORDER BY id DESC LIMIT 20"
            )
        rows = await cursor.fetchall()

    # Build nodes and edges from entity co-occurrence
    entity_map = {}  # name -> {type, role, count}
    edge_counts = defaultdict(int)  # (name1, name2) -> count

    for row in rows:
        entities = json.loads(row["entities"]) if row["entities"] else []
        names = []
        for ent in entities:
            name = ent.get("name", "")
            if not name:
                continue
            if name not in entity_map:
                entity_map[name] = {
                    "type": ent.get("type", "unknown"),
                    "role": ent.get("role", ""),
                    "count": 0,
                }
            entity_map[name]["count"] += 1
            names.append(name)

        # Co-occurrence edges
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = tuple(sorted([names[i], names[j]]))
                edge_counts[key] += 1

    nodes = [
        {"id": name, "type": info["type"], "role": info["role"], "count": info["count"]}
        for name, info in entity_map.items()
    ]
    edges = [
        {"source": k[0], "target": k[1], "weight": v}
        for k, v in edge_counts.items()
    ]

    return {"nodes": nodes, "edges": edges}
