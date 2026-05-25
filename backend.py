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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
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


def db_execute(query: str, params: tuple = ()) -> int:
    """
    Run a write query (INSERT / UPDATE / DELETE) and return the number of
    affected rows.  Uses PostgreSQL when available, falls back to SQLite.
    Write queries with ? placeholders; they are rewritten to %s for PostgreSQL.
    """
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        if params:
            cur.execute(query.replace("?", "%s"), params)
        else:
            cur.execute(query)
        affected = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return affected
    else:
        conn = sqlite3.connect(DB_PATH)
        if params:
            cur = conn.execute(query, params)
        else:
            cur = conn.execute(query)
        conn.commit()
        affected = cur.rowcount
        conn.close()
        return affected


# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/api/sentiment")
def sentiment_summary(
    date_from: Optional[str] = Query(None, description="Start date YYYY-MM-DD; filters by published date"),
    date_to:   Optional[str] = Query(None, description="End date YYYY-MM-DD (inclusive); filters by published date"),
):
    """
    Per-niche sentiment aggregates.
    Default (no dates): aggregates headlines fetched in the last 24 hours.
    With date_from / date_to: aggregates headlines whose published date falls
    within the given range (inclusive on both ends).
    Returns avg compound/pos/neg scores, total headline count, and per-label
    counts (count_pos / count_neg / count_neu) for each niche.
    Sorted by avg_compound descending (most bullish first).
    """
    _SQL = """
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
        WHERE {where}
        GROUP BY niche
        ORDER BY avg_compound DESC
    """

    if date_from or date_to:
        conditions: list[str] = []
        params: list[str]     = []
        if date_from:
            conditions.append("published >= ?")
            params.append(date_from)
        if date_to:
            # Add one day so the entire to-date is included
            try:
                next_day = (date.fromisoformat(date_to) + timedelta(days=1)).isoformat()
            except ValueError:
                next_day = date_to
            conditions.append("published < ?")
            params.append(next_day)
        rows = db_query(_SQL.format(where=" AND ".join(conditions)), tuple(params))
    else:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        rows = db_query(_SQL.format(where="fetched >= ?"), (cutoff,))

    # Round in Python so the same query works for both SQLite and PostgreSQL
    for row in rows:
        row["avg_compound"] = round(row["avg_compound"] or 0, 4)
        row["avg_pos"]      = round(row["avg_pos"]      or 0, 4)
        row["avg_neg"]      = round(row["avg_neg"]      or 0, 4)
    return rows


@app.get("/api/headlines")
def get_headlines(
    niche:     Optional[str] = Query(None, description="Filter by niche name"),
    date_from: Optional[str] = Query(None, description="Start date YYYY-MM-DD; filters by published date"),
    date_to:   Optional[str] = Query(None, description="End date YYYY-MM-DD (inclusive); filters by published date"),
):
    """
    Return the 20 most recent headlines.
    Pass ?niche=<name> to restrict results to a single niche.
    Pass ?date_from=YYYY-MM-DD and/or ?date_to=YYYY-MM-DD to restrict by
    published date (inclusive on both ends).
    All filters are combinable; omitting all gives the 20 most recent overall.
    """
    conditions: list[str] = []
    params: list[str]     = []

    if niche:
        conditions.append("niche = ?")
        params.append(niche)

    if date_from:
        conditions.append("SUBSTR(published, 1, 10) >= ?")
        params.append(date_from)

    if date_to:
        conditions.append("SUBSTR(published, 1, 10) <= ?")
        params.append(date_to)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return db_query(
        f"""
        SELECT niche, title, source, url, published, compound, pos, neg
        FROM headlines
        {where}
        ORDER BY fetched DESC, published DESC
        LIMIT 20
        """,
        tuple(params),
    )


_CLEANUP_KEY = "cleanup123"


@app.get("/api/cleanup")
def cleanup_malformed_dates(
    key: Optional[str] = Query(None, description="Secret key required to run cleanup"),
):
    """
    Delete all headlines whose published value does NOT start with '20'
    (i.e. old RFC-2822 / non-ISO dates that slipped through before normalisation).
    Requires ?key=cleanup123.
    Returns { "deleted": <count> }.
    """
    if key != _CLEANUP_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing key.")

    deleted = db_execute(
        "DELETE FROM headlines WHERE published NOT LIKE '20%'", ()
    )
    return {"deleted": deleted}


# ── entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
