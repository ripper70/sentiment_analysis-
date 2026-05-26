"""
fetch_news.py
Pull headlines from NewsAPI + Reuters/AP RSS feeds, score with local FinBERT
(ProsusAI/finbert via transformers), store in DB.  Falls back to VADER on error.
Uses PostgreSQL when DATABASE_URL env var is set, otherwise falls back to SQLite.
Run:  python3 fetch_news.py
"""

import email.utils
import os
import sqlite3
import time
from datetime import datetime, timezone

import dateutil.parser
import feedparser
import requests
from newsapi import NewsApiClient
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import NEWS_API_KEY, NICHES

# ── db config ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_PATH      = "sentiment.db"
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2

api      = NewsApiClient(api_key=NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()   # VADER — used as fallback only

# ── FinBERT local model (lazy-loaded on first call) ────────────────────────────
_finbert_pipeline = None

RSS_FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
    "https://feeds.a.dj.com/rss/RSSWSJD.xml",
    "https://finance.yahoo.com/news/rssindex",
    "https://www.investing.com/rss/news.rss",
    "https://fortune.com/feed/",
    "https://axios.com/feeds/feed.rss",
]

GOSSIP_BLOCKLIST = [
    "wedding dress", "fashion tips", "recipe", "horoscope", "celebrity gossip",
    "romance", "dating tips", "divorce", "pottery class", "dance class",
    "yoga tips", "beauty tips", "makeup", "skincare routine", "best dressed",
    "outfit ideas",
]


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


# ── date normalisation ─────────────────────────────────────────────────────────

def normalize_date(raw: str) -> str | None:
    """Parse an RSS/NewsAPI date string and return an ISO-8601 string, or None."""
    if not raw:
        return None
    # Primary: RFC 2822 format used by most RSS feeds
    # e.g. "Wed, 20 May 2026 21:56:06 +0000"
    try:
        return email.utils.parsedate_to_datetime(raw).isoformat()
    except Exception:
        pass
    # Fallback: dateutil handles ISO 8601, NewsAPI timestamps, and other variants
    try:
        return dateutil.parser.parse(raw).isoformat()
    except Exception:
        pass
    return None


# ── scoring ────────────────────────────────────────────────────────────────────

def _get_finbert():
    """Return the FinBERT pipeline, loading it on the first call."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        _finbert_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
    return _finbert_pipeline


def score_finbert(text: str) -> dict:
    """Score a headline with local FinBERT; falls back to VADER on error.

    Returns a dict with keys: compound, pos, neg, neu — same shape as VADER so
    the rest of the pipeline needs no changes.

    Label mapping:
      positive → compound= score, pos=score, neg=0,     neu=0
      negative → compound=-score, pos=0,     neg=score, neu=0
      neutral  → compound= 0,     pos=0,     neg=0,     neu=score
    """
    try:
        finbert = _get_finbert()
        result  = finbert(text)[0]
        label   = result["label"].lower()
        s       = result["score"]
        if label == "positive":
            return {"compound":  s,   "pos": s,   "neg": 0.0, "neu": 0.0}
        elif label == "negative":
            return {"compound": -s,   "pos": 0.0, "neg": s,   "neu": 0.0}
        else:  # neutral
            return {"compound":  0.0, "pos": 0.0, "neg": 0.0, "neu": s}
    except Exception as e:
        print(f"    [!] FinBERT error — falling back to VADER: {e}")
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


def fetch_all_rss() -> list[dict]:
    """Fetch every RSS feed and return a flat pool of article dicts."""
    pool = []
    for url in RSS_FEEDS:
        try:
            feed   = feedparser.parse(url)
            source = feed.feed.get("title", url) if feed.feed else url
            for entry in feed.entries:
                title = (entry.get("title") or "").strip()
                if not title:
                    continue
                pool.append({
                    "title":     title,
                    "source":    source,
                    "url":       entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
        except Exception as e:
            print(f"    [!] RSS — {url}: {e}")
    return pool


def fetch_rss(niche: str, keywords: list[str], pool: list[dict]) -> list[dict]:
    """Filter the pre-fetched RSS pool for articles matching any niche keyword."""
    lower_kws = [kw.lower() for kw in keywords]
    return [
        a for a in pool
        if any(kw in a["title"].lower() for kw in lower_kws)
    ]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    conn       = get_conn()
    init_db(conn)
    fetched_at = datetime.now(timezone.utc).isoformat()

    # all_articles collects unique articles across niches before scoring
    all_articles: list[dict]     = []
    niche_counts: dict[str, int] = {}

    print(f"Fetching from NewsAPI + RSS feeds for {len(NICHES)} niches…\n")

    # ── fetch all RSS articles once into a shared pool ────────────────────────
    print("  Pulling RSS feeds…")
    rss_pool = fetch_all_rss()
    print(f"  RSS pool: {len(rss_pool)} total articles from {len(RSS_FEEDS)} feeds\n")

    # ── global dedup: same article must not appear in more than one niche ─────
    # Keyed by (normalised_title, source) so the same story from two outlets is
    # allowed, but a single article matched by two different niche keywords is
    # saved only once (under whichever niche processes it first).
    seen_global: set[tuple[str, str]] = set()

    for niche, keywords in NICHES.items():
        # ── pull from both sources ────────────────────────────────────────────
        news_articles = fetch_newsapi(niche, keywords)
        rss_articles  = fetch_rss(niche, keywords, rss_pool)

        combined = news_articles + rss_articles

        # ── quality filters ───────────────────────────────────────────────────
        combined = [a for a in combined if len(a["title"]) >= 25]
        combined = [
            a for a in combined
            if not any(word in a["title"].lower() for word in GOSSIP_BLOCKLIST)
        ]

        # ── deduplicate by normalised title (intra-niche) and by
        #    (title, source) across all niches (inter-niche) ─────────────────
        seen:   set[str]   = set()
        unique: list[dict] = []
        for a in combined:
            key        = a["title"].lower()
            global_key = (key, a["source"])
            if key not in seen and global_key not in seen_global:
                seen.add(key)
                seen_global.add(global_key)
                unique.append({**a, "niche": niche})

        niche_counts[niche] = len(unique)
        all_articles.extend(unique)

        news_n    = len(news_articles)
        rss_n     = len(rss_articles)
        raw_total = news_n + rss_n
        filtered  = raw_total - len(combined)
        deduped   = len(combined) - len(unique)
        note_parts = []
        if filtered:
            note_parts.append(f"{filtered} filtered")
        if deduped:
            note_parts.append(f"{deduped} dupes removed")
        note = (", " + ", ".join(note_parts)) if note_parts else ""
        print(
            f"  ✓  {niche:<25s} — {len(unique):3d} headlines "
            f"(NewsAPI: {news_n}, RSS: {rss_n}{note})"
        )

    # ── score all headlines with FinBERT in batches of 10 ────────────────────
    BATCH_SIZE = 10
    total = len(all_articles)
    rows:  list[tuple] = []

    print(f"\nScoring {total} headlines with FinBERT (ProsusAI/finbert)…")
    for i in range(0, total, BATCH_SIZE):
        batch     = all_articles[i : i + BATCH_SIZE]
        batch_end = min(i + BATCH_SIZE, total)
        print(f"  Scoring headlines {i + 1}–{batch_end} of {total}…")
        for a in batch:
            sc = score_finbert(a["title"])
            rows.append((
                a["niche"],
                a["title"],
                a["source"],
                a["url"],
                normalize_date(a["published"]),
                sc["compound"], sc["pos"], sc["neu"], sc["neg"],
                fetched_at,
            ))
            time.sleep(0.1)

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
