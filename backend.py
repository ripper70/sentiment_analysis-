"""
backend.py — FastAPI server for the Sentiment Pulse Dashboard.

Usage:
    python3 backend.py
    # or: uvicorn backend:app --reload --port 8000

Endpoints:
    GET /api/sentiment   — per-niche aggregates for the last 24 hours
    GET /api/headlines   — 20 most recent headlines; ?niche= filters to one niche
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# ── config ─────────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "sentiment.db"

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sentiment Pulse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── db helper ──────────────────────────────────────────────────────────────────
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/api/sentiment")
def sentiment_summary():
    """
    Per-niche sentiment aggregates for headlines fetched in the last 24 hours.
    Returns avg compound/pos/neg scores, total headline count, and per-label
    counts (count_pos / count_neg / count_neu) for each niche.
    Sorted by avg_compound descending (most bullish first).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT
            niche,
            ROUND(AVG(compound), 4)  AS avg_compound,
            ROUND(AVG(pos),      4)  AS avg_pos,
            ROUND(AVG(neg),      4)  AS avg_neg,
            COUNT(*)                 AS count,
            SUM(CASE WHEN compound >=  0.05 THEN 1 ELSE 0 END) AS count_pos,
            SUM(CASE WHEN compound <= -0.05 THEN 1 ELSE 0 END) AS count_neg,
            SUM(CASE WHEN compound >  -0.05
                      AND compound <   0.05 THEN 1 ELSE 0 END) AS count_neu
        FROM headlines
        WHERE fetched >= ?
        GROUP BY niche
        ORDER BY avg_compound DESC
        """,
        (cutoff,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/headlines")
def get_headlines(
    niche: Optional[str] = Query(None, description="Filter by niche name"),
):
    """
    Return the 20 most recent headlines.
    Pass ?niche=<name> to restrict results to a single niche;
    omit the parameter to get the 20 most recent across all niches.
    """
    conn = get_conn()
    if niche:
        rows = conn.execute(
            """
            SELECT niche, title, source, url, published, compound, pos, neg
            FROM headlines
            WHERE niche = ?
            ORDER BY fetched DESC, published DESC
            LIMIT 20
            """,
            (niche,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT niche, title, source, url, published, compound, pos, neg
            FROM headlines
            ORDER BY fetched DESC, published DESC
            LIMIT 20
            """
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
