"""
rescore_finbert.py
Re-score every headline already stored in PostgreSQL with FinBERT.

For each row the script tries to fetch the full article body text first
(richer signal); it falls back to the headline title when the fetch fails.
Scores are written back to the compound, pos, neg, neu columns.

Usage:
    python3 rescore_finbert.py

Environment:
    DATABASE_URL — PostgreSQL connection string (set by Railway automatically).
                   Falls back to FALLBACK_DATABASE_URL below if not set.
"""

import os

import psycopg2

from fetch_news import fetch_article_text, score_finbert

# ── connection config ──────────────────────────────────────────────────────────
# Paste your Railway "Public URL" here as a safety fallback.
FALLBACK_DATABASE_URL = (
    "postgresql://postgres:VLdBgQlXTPcYcxKeDigPjxLZdmvksOco@ballast.proxy.rlwy.net:51402/railway"
)

DATABASE_URL = os.environ.get("DATABASE_URL") or FALLBACK_DATABASE_URL

BATCH_SIZE = 50


# ── helpers ────────────────────────────────────────────────────────────────────

def get_conn():
    """Return an open psycopg2 connection."""
    return psycopg2.connect(DATABASE_URL)


def fetch_all_headlines(conn):
    """Return every headline row as a list of (id, title, url) tuples."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, url FROM headlines ORDER BY id")
        return cur.fetchall()


def update_scores(conn, rows: list[tuple]):
    """Bulk-update sentiment columns for a batch of rows.

    Each element of *rows* must be (compound, pos, neg, neu, id).
    """
    sql = """
        UPDATE headlines
           SET compound = %s,
               pos      = %s,
               neg      = %s,
               neu      = %s
         WHERE id = %s
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Connecting to PostgreSQL…")
    conn = get_conn()

    print("Fetching all headlines…")
    headlines = fetch_all_headlines(conn)
    total = len(headlines)
    print(f"Found {total} headlines to re-score.\n")

    updated   = 0
    batch_buf = []   # accumulates (compound, pos, neg, neu, id) tuples

    for i, (row_id, title, url) in enumerate(headlines, start=1):
        # ── score ──────────────────────────────────────────────────────────────
        text = None
        if url:
            text = fetch_article_text(url)
        if not text:
            text = title

        sc = score_finbert(text)
        batch_buf.append((sc["compound"], sc["pos"], sc["neg"], sc["neu"], row_id))

        # ── flush batch ────────────────────────────────────────────────────────
        if len(batch_buf) >= BATCH_SIZE:
            update_scores(conn, batch_buf)
            updated += len(batch_buf)
            batch_buf.clear()
            pct = updated / total * 100
            print(f"  Progress: {updated}/{total} ({pct:.1f}%) updated")

    # ── flush remainder ────────────────────────────────────────────────────────
    if batch_buf:
        update_scores(conn, batch_buf)
        updated += len(batch_buf)
        batch_buf.clear()

    conn.close()
    print(f"\n✅  Done — {updated}/{total} headlines re-scored with FinBERT.")


if __name__ == "__main__":
    main()
