"""
fetch_news.py
Pull headlines from NewsAPI + The Guardian, score with VADER, store in DB.
Uses PostgreSQL when DATABASE_URL env var is set, otherwise falls back to SQLite.
Run:  python3 fetch_news.py
"""

import os
import sqlite3
import time
from datetime import datetime, timezone

import requests
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import GUARDIAN_API_KEY, NEWS_API_KEY, NICHES

# ── db config ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_PATH      = "sentiment.db"
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2

api      = NewsApiClient(api_key=NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()

GUARDIAN_URL = "https://content.guardianapis.com/search"


# ── database setup ─────────────────────────────────────────────────────────────

def get_conn():
    """Return a database connection (PostgreSQL or SQLite)."""
    if USE_POSTGRES:
        return psycopg2.connect(DATABASE_URL)
    return sqlite3.connect(DB_PATH)


def init_db(conn):
    """Create the headlines table and indexes if they don't exist."""
    # id column syntax differs between PostgreSQL and SQLite
    id_col = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS headlines (
            id        {id_col},
            niche     TEXT    NOT NULL,
            title     TEXT    NOT NULL,
            source    TEXT,
            url       TEXT,
            published TEXT,
            compound  REAL,
            pos       REAL,
            neu       REAL,
            neg       REAL,
            fetched   TEXT    NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_niche   ON headlines(niche)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fetched ON headlines(fetched)")
    conn.commit()


# ── scoring ────────────────────────────────────────────────────────────────────

def score(text: str) -> dict:
    return analyzer.polarity_scores(text)


# ── fetchers ───────────────────────────────────────────────────────────────────

def fetch_newsapi(niche: str, keywords: list[str]) -> list[dict]:
    """Return raw article dicts from NewsAPI for every keyword in a niche."""
    articles = []
    for keyword in keywords:
        try:
            result = api.get_everything(
                q=keyword,
                language="en",
                sort_by="publishedAt",
                page_size=20,
            )
            for a in result.get("articles", []):
                title = (a.get("title") or "").strip()
                if not title or title == "[Removed]":
                    continue
                articles.append({
                    "title":     title,
                    "source":    (a.get("source") or {}).get("name", "NewsAPI"),
                    "url":       a.get("url", ""),
                    "published": a.get("publishedAt", ""),
                })
            time.sleep(0.25)   # stay within rate limit
        except Exception as e:
            print(f"    [!] NewsAPI — {niche} / '{keyword}': {e}")
    return articles


def fetch_guardian(niche: str, keywords: list[str]) -> list[dict]:
    """Return raw article dicts from The Guardian for every keyword in a niche."""
    articles = []
    for keyword in keywords:
        try:
            resp = requests.get(
                GUARDIAN_URL,
                params={
                    "api-key":     GUARDIAN_API_KEY,
                    "q":           keyword,
                    "show-fields": "headline",
                    "page-size":   20,
                    "order-by":    "newest",
                },
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json().get("response", {}).get("results", [])
            for r in results:
                # Prefer the richer headline field; fall back to webTitle
                fields = r.get("fields") or {}
                title  = (fields.get("headline") or r.get("webTitle") or "").strip()
                if not title:
                    continue
                articles.append({
                    "title":     title,
                    "source":    "The Guardian",
                    "url":       r.get("webUrl", ""),
                    "published": r.get("webPublicationDate", ""),
                })
            time.sleep(0.25)
        except Exception as e:
            print(f"    [!] Guardian — {niche} / '{keyword}': {e}")
    return articles


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    conn       = get_conn()
    init_db(conn)
    fetched_at = datetime.now(timezone.utc).isoformat()

    rows: list[tuple]           = []
    niche_counts: dict[str, int] = {}

    print(f"Fetching from NewsAPI + The Guardian for {len(NICHES)} niches…\n")

    for niche, keywords in NICHES.items():
        # ── pull from both sources ────────────────────────────────────────────
        news_articles     = fetch_newsapi(niche, keywords)
        guardian_articles = fetch_guardian(niche, keywords)

        combined = news_articles + guardian_articles

        # ── deduplicate by normalised title ───────────────────────────────────
        seen:   set[str]   = set()
        unique: list[dict] = []
        for a in combined:
            key = a["title"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)

        # ── score & collect rows ──────────────────────────────────────────────
        for a in unique:
            sc = score(a["title"])
            rows.append((
                niche,
                a["title"],
                a["source"],
                a["url"],
                a["published"],
                sc["compound"], sc["pos"], sc["neu"], sc["neg"],
                fetched_at,
            ))

        niche_counts[niche] = len(unique)
        news_n     = len(news_articles)
        guardian_n = len(guardian_articles)
        deduped    = news_n + guardian_n - len(unique)
        print(
            f"  ✓  {niche:<25s} — {len(unique):3d} headlines "
            f"(NewsAPI: {news_n}, Guardian: {guardian_n}"
            + (f", {deduped} dupes removed)" if deduped else ")")
        )

    # ── bulk insert ───────────────────────────────────────────────────────────
    placeholder = "%s" if USE_POSTGRES else "?"
    insert_sql  = f"""
        INSERT INTO headlines
          (niche, title, source, url, published, compound, pos, neu, neg, fetched)
        VALUES ({', '.join([placeholder] * 10)})
    """
    cur = conn.cursor()
    cur.executemany(insert_sql, rows)
    conn.commit()
    cur.close()
    conn.close()

    db_label = DATABASE_URL.split("@")[-1] if USE_POSTGRES else DB_PATH
    print(f"\n✅  Done — {len(rows)} headlines saved to {db_label}")


if __name__ == "__main__":
    main()
