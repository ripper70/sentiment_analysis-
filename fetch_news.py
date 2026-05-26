"""
fetch_news.py
Pull headlines from NewsAPI + Reuters/AP RSS feeds, classify each article
into one of 16 niches using facebook/bart-large-mnli (zero-shot NLI), score
sentiment with local FinBERT (ProsusAI/finbert via transformers), store in DB.
Falls back to VADER on FinBERT error; falls back to "Politics & Economy" on
classifier error.
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
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import NEWS_API_KEY

# ── db config ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_PATH      = "sentiment.db"
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2

api      = NewsApiClient(api_key=NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()   # VADER — used as fallback only

# ── lazy-loaded model handles ──────────────────────────────────────────────────
_finbert_pipeline    = None   # ProsusAI/finbert  (sentiment scoring)
_classifier_pipeline = None   # facebook/bart-large-mnli  (niche classification)

# ── canonical niche labels ─────────────────────────────────────────────────────
NICHE_LABELS: list[str] = [
    "Energy & Oil",
    "Real Estate",
    "Banking & Finance",
    "Food & Agriculture",
    "Retail & E-Commerce",
    "Social Media & AdTech",
    "Travel & Tourism",
    "Gaming & Esports",
    "Healthcare & Biotech",
    "Semiconductors",
    "Cybersecurity",
    "Electric Vehicles",
    "Bitcoin & Crypto",
    "Space Technology",
    "AI & Machine Learning",
    "Politics & Economy",
]

# ── broad financial keywords for NewsAPI sweep ─────────────────────────────────
BROAD_KEYWORDS: list[str] = [
    "market", "economy", "stocks", "finance", "technology",
    "earnings", "trade", "inflation", "Fed", "GDP",
]

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
    try:
        return email.utils.parsedate_to_datetime(raw).isoformat()
    except Exception:
        pass
    try:
        return dateutil.parser.parse(raw).isoformat()
    except Exception:
        pass
    return None


# ── FinBERT sentiment scoring ──────────────────────────────────────────────────

def _get_finbert():
    """Return the FinBERT pipeline, loading it on the first call."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        _finbert_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
    return _finbert_pipeline


def score_finbert(text: str) -> dict:
    """Score a headline with local FinBERT; falls back to VADER on error.

    Prepends an investor-framing prefix so FinBERT interprets ambiguous text
    through an economic/market lens rather than a generic financial-tone lens.

    Returns a dict with keys: compound, pos, neg, neu — same shape as VADER.

    Label mapping:
      positive → compound= score, pos=score, neg=0,     neu=0
      negative → compound=-score, pos=0,     neg=score, neu=0
      neutral  → compound= 0,     pos=0,     neg=0,     neu=score
    """
    try:
        finbert = _get_finbert()
        framed  = "How does this news impact consumer investments and financial markets? " + text
        result  = finbert(framed)[0]
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


# ── zero-shot niche classifier ─────────────────────────────────────────────────

def _get_classifier():
    """Return the zero-shot classification pipeline, loading it on first call."""
    global _classifier_pipeline
    if _classifier_pipeline is None:
        print("  [classifier] Loading facebook/bart-large-mnli (first run only)…")
        _classifier_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _classifier_pipeline


def classify_niche(text: str) -> str:
    """Classify article text into one of the 16 canonical niches.

    Uses the first 512 characters of *text* as input to the zero-shot
    facebook/bart-large-mnli model.  Falls back to 'Politics & Economy'
    if the classifier raises an exception or *text* is empty.
    """
    try:
        snippet = (text or "")[:512].strip()
        if not snippet:
            return "Politics & Economy"
        framed_snippet = "Classify this news article into the most relevant financial and economic category: " + snippet
        classifier = _get_classifier()
        result     = classifier(framed_snippet, NICHE_LABELS)
        return result["labels"][0]          # highest-scoring label
    except Exception as e:
        print(f"    [!] Classifier error — defaulting to Politics & Economy: {e}")
        return "Politics & Economy"


# ── article text fetcher ──────────────────────────────────────────────────────

def fetch_article_text(url: str) -> str | None:
    """Fetch full article body text for richer sentiment/classification scoring.

    Makes a browser-like GET request, extracts all <p> tag text via
    BeautifulSoup, and returns the first 512 characters (FinBERT's practical
    max input length).  Returns None when:
      - the request fails or times out (5 s limit)
      - the extracted text is shorter than 100 characters
      - any other exception occurs
    Never raises — all errors are swallowed silently.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text = " ".join(p.get_text() for p in soup.find_all("p")).strip()
        if len(text) < 100:
            return None
        return text[:512]
    except Exception:
        return None


# ── fetchers ───────────────────────────────────────────────────────────────────

def fetch_newsapi(label: str, keywords: list[str]) -> list[dict]:
    """Return raw article dicts from NewsAPI for every keyword in *keywords*.

    *label* is used only for error-message context (e.g. 'broad financial').
    """
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
            print(f"    [!] NewsAPI — {label} / '{keyword}': {e}")
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


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    conn       = get_conn()
    init_db(conn)
    fetched_at = datetime.now(timezone.utc).isoformat()

    print("Fetching articles from NewsAPI (broad financial keywords) + RSS feeds…\n")

    # ── 1. Pull all RSS articles into a flat list ─────────────────────────────
    print("  Pulling RSS feeds…")
    rss_articles = fetch_all_rss()
    print(f"  RSS pool: {len(rss_articles)} articles from {len(RSS_FEEDS)} feeds\n")

    # ── 2. Pull NewsAPI with broad financial keywords ─────────────────────────
    print("  Pulling NewsAPI with broad financial keywords…")
    news_articles = fetch_newsapi("broad financial", BROAD_KEYWORDS)
    print(f"  NewsAPI: {len(news_articles)} articles\n")

    # ── 3. Combine, apply quality filters, and deduplicate globally ───────────
    combined = rss_articles + news_articles

    combined = [a for a in combined if len(a["title"]) >= 25]
    combined = [
        a for a in combined
        if not any(word in a["title"].lower() for word in GOSSIP_BLOCKLIST)
    ]

    seen_global: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for a in combined:
        key        = a["title"].lower()
        global_key = (key, a["source"])
        if global_key not in seen_global:
            seen_global.add(global_key)
            unique.append(a)

    filtered = len(rss_articles) + len(news_articles) - len(combined)
    duped    = len(combined) - len(unique)
    print(
        f"  Combined: {len(rss_articles) + len(news_articles)} articles  "
        f"→  {filtered} filtered  →  {duped} dupes removed  "
        f"→  {len(unique)} unique\n"
    )

    # ── 4. Classify niche, score sentiment, collect DB rows ───────────────────
    print(f"Classifying niches + scoring {len(unique)} articles with FinBERT…\n")

    niche_counts: dict[str, int] = {}
    rows: list[tuple] = []

    for i, a in enumerate(unique, 1):
        text  = fetch_article_text(a["url"]) or a["title"]
        niche = classify_niche(text)
        sc    = score_finbert(text)

        niche_counts[niche] = niche_counts.get(niche, 0) + 1

        rows.append((
            niche,
            a["title"],
            a["source"],
            a["url"],
            normalize_date(a["published"]),
            sc["compound"], sc["pos"], sc["neu"], sc["neg"],
            fetched_at,
        ))

        title_preview = a["title"][:70] + "…" if len(a["title"]) > 70 else a["title"]
        print(f"  [{i:3d}/{len(unique)}] {title_preview}")
        print(f"           → {niche}")

        time.sleep(0.05)

    # ── 5. Bulk insert ─────────────────────────────────────────────────────────
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

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    db_label = DATABASE_URL.split("@")[-1] if USE_POSTGRES else DB_PATH
    print(f"\n✅  Done — {len(rows)} headlines saved to {db_label}")
    print("\n── Niche breakdown ──────────────────────────────────────────────────")
    for niche in NICHE_LABELS:
        count = niche_counts.get(niche, 0)
        if count:
            bar = "█" * min(count, 50)
            print(f"  {niche:<25s} {count:3d}  {bar}")
    print("─────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
