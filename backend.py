"""
backend.py — FastAPI server for the Sentiment Pulse Dashboard.

Usage:
    python3 backend.py
    # or: uvicorn backend:app --reload --port 8000

Endpoints:
    GET /api/sentiment   — per-niche aggregates for the last 24 hours
    GET /api/headlines   — 20 most recent headlines; ?niche= filters to one niche
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# ── config ─────────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_PATH      = Path(__file__).parent / "sentiment.db"
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sentiment Pulse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── db helpers ─────────────────────────────────────────────────────────────────
def db_query(query: str, params: tuple = ()) -> list[dict]:
    """
    Run a SELECT query and return rows as a list of dicts.
    Uses PostgreSQL (DATABASE_URL) when available, falls back to SQLite.
    Write queries with ? placeholders; they are rewritten to %s for PostgreSQL.
    """
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(query.replace("?", "%s"), params)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
        return rows
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        conn.close()
        return rows


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
    rows = db_query(
        """
        SELECT
            niche,
            AVG(compound)  AS avg_compound,
            AVG(pos)       AS avg_pos,
            AVG(neg)       AS avg_neg,
            COUNT(*)       AS count,
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
    )
    # Round in Python so the same query works for both SQLite and PostgreSQL
    for row in rows:
        row["avg_compound"] = round(row["avg_compound"] or 0, 4)
        row["avg_pos"]      = round(row["avg_pos"]      or 0, 4)
        row["avg_neg"]      = round(row["avg_neg"]      or 0, 4)
    return rows


@app.get("/api/headlines")
def get_headlines(
    niche: Optional[str] = Query(None, description="Filter by niche name"),
):
    """
    Return the 20 most recent headlines.
    Pass ?niche=<name> to restrict results to a single niche;
    omit the parameter to get the 20 most recent across all niches.
    """
    if niche:
        return db_query(
            """
            SELECT niche, title, source, url, published, compound, pos, neg
            FROM headlines
            WHERE niche = ?
            ORDER BY fetched DESC, published DESC
            LIMIT 20
            """,
            (niche,),
        )
    return db_query(
        """
        SELECT niche, title, source, url, published, compound, pos, neg
        FROM headlines
        ORDER BY fetched DESC, published DESC
        LIMIT 20
        """
    )


# ── entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
