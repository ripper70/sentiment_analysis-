"""
fetch_news.py
Pull headlines from NewsAPI, score with VADER, store in SQLite.
Run:  python3 fetch_news.py
"""

import sqlite3, time
from datetime import datetime, timezone

from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import NEWS_API_KEY, NICHES

DB_PATH = "sentiment.db"
api      = NewsApiClient(api_key=NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()


# ── database setup ────────────────────────────────────────────────────────────

def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS headlines (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_niche    ON headlines(niche)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fetched  ON headlines(fetched)")
    conn.commit()


# ── scoring ───────────────────────────────────────────────────────────────────

def score(text: str) -> dict:
    return analyzer.polarity_scores(text)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    conn       = sqlite3.connect(DB_PATH)
    init_db(conn)
    fetched_at = datetime.now(timezone.utc).isoformat()

    rows: list[tuple] = []
    niche_counts: dict[str, int] = {}

    print(f"Fetching from NewsAPI for {len(NICHES)} niches…\n")

    for niche, keywords in NICHES.items():
        niche_total = 0
        for keyword in keywords:
            try:
                result = api.get_everything(
                    q=keyword,
                    language="en",
                    sort_by="publishedAt",
                    page_size=20,
                )
                articles = result.get("articles", [])
                for a in articles:
                    title = (a.get("title") or "").strip()
                    if not title or title == "[Removed]":
                        continue
                    source    = (a.get("source") or {}).get("name", "")
                    url       = a.get("url", "")
                    published = a.get("publishedAt", "")
                    sc        = score(title)
                    rows.append((
                        niche, title, source, url, published,
                        sc["compound"], sc["pos"], sc["neu"], sc["neg"],
                        fetched_at,
                    ))
                    niche_total += 1
                time.sleep(0.25)   # stay well within rate limits
            except Exception as e:
                print(f"  [!] {niche} / '{keyword}': {e}")

        niche_counts[niche] = niche_total
        print(f"  ✓  {niche:<25s} — {niche_total:3d} headlines")

    conn.executemany("""
        INSERT INTO headlines
          (niche, title, source, url, published, compound, pos, neu, neg, fetched)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    conn.close()

    print(f"\n✅  Done — {len(rows)} headlines saved to {DB_PATH}")


if __name__ == "__main__":
    main()
