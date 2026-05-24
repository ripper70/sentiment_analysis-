"""
fetch_news.py
Pull headlines for all 15 niches from NewsAPI, score with VADER, store in SQLite.
Run:  python3 fetch_news.py
"""

import sqlite3, requests, json, time
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import NEWS_API_KEY, NICHES

DB_PATH = "sentiment.db"
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_niche ON headlines(niche)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fetched ON headlines(fetched)")
    conn.commit()


# ── NewsAPI fetch ─────────────────────────────────────────────────────────────

def fetch_headlines(query: str, page_size: int = 20) -> list[dict]:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": page_size,
        "apiKey":   NEWS_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("articles", [])
    except Exception as e:
        print(f"  [!] API error for '{query}': {e}")
        return []


# ── scoring ───────────────────────────────────────────────────────────────────

def score(text: str) -> dict:
    return analyzer.polarity_scores(text)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    fetched_at = datetime.now(timezone.utc).isoformat()

    total = 0
    for niche, keywords in NICHES.items():
        query = " OR ".join(f'"{kw}"' for kw in keywords[:2])   # use top-2 keywords
        print(f"Fetching: {niche} …", end=" ", flush=True)
        articles = fetch_headlines(query)

        rows = []
        for a in articles:
            title = (a.get("title") or "").strip()
            if not title or title == "[Removed]":
                continue
            sc = score(title)
            rows.append((
                niche,
                title,
                a.get("source", {}).get("name"),
                a.get("url"),
                a.get("publishedAt"),
                sc["compound"],
                sc["pos"],
                sc["neu"],
                sc["neg"],
                fetched_at,
            ))

        conn.executemany("""
            INSERT INTO headlines
              (niche, title, source, url, published, compound, pos, neu, neg, fetched)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, rows)
        conn.commit()
        print(f"{len(rows)} headlines saved.")
        total += len(rows)
        time.sleep(0.3)   # be polite to the API

    conn.close()
    print(f"\n✅  Done — {total} total headlines saved to {DB_PATH}")


if __name__ == "__main__":
    main()
