"""
api.py  —  read sentiment.db and return JSON for the dashboard
Usage:    python3 api.py          (starts server on port 8000)
Requires: pip install fastapi uvicorn
"""

import sqlite3, json
from pathlib import Path
from collections import defaultdict

DB_PATH = Path(__file__).parent / "sentiment.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def niche_summary():
    conn = get_conn()
    rows = conn.execute("""
        SELECT niche,
               AVG(compound)  AS avg_compound,
               AVG(pos)       AS avg_pos,
               AVG(neg)       AS avg_neg,
               COUNT(*)       AS count
        FROM headlines
        GROUP BY niche
        ORDER BY avg_compound DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def niche_headlines(niche: str, limit: int = 10):
    conn = get_conn()
    rows = conn.execute("""
        SELECT title, source, compound, pos, neg, published
        FROM headlines
        WHERE niche = ?
        ORDER BY published DESC
        LIMIT ?
    """, (niche, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    # Quick CLI test
    summaries = niche_summary()
    print(json.dumps(summaries, indent=2))
