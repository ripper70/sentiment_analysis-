"""
rescore_finbert.py
Re-score every headline already stored in PostgreSQL with FinBERT.

For each row the script tries to fetch the full article body text first
(richer signal); it falls back to the headline title when the fetch fails.
Scores are written back to the compound, pos, neg, neu columns.

Usage:
    python3 rescore_finbert.py

Environment:
    DATABASE_URL — PostgreSQL connection string (required; set by Railway
                   automatically, or export it locally before running).
"""

import os

import psycopg2

from fetch_news import classify_niche, fetch_article_text, score_finbert

# ── connection config ──────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set")
    exit(1)

BATCH_SIZE = 50


# ── helpers ────────────────────────────────────────────────────────────────────

def get_conn():
    """Return an open psycopg2 connection."""
    return psycopg2.connect(DATABASE_URL)


def fetch_all_headlines(conn):
    """Return every headline row as a list of (id, title, url, niche) tuples."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, url, niche FROM headlines ORDER BY id")
        return cur.fetchall()


def update_scores(conn, rows: list[tuple]):
    """Bulk-update sentiment columns and niche for a batch of rows.

    Each element of *rows* must be (compound, pos, neg, neu, niche, id).
    """
    sql = """
        UPDATE headlines
           SET compound = %s,
               pos      = %s,
               neg      = %s,
               neu      = %s,
               niche    = %s
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
    batch_buf = []   # accumulates (compound, pos, neg, neu, niche, id) tuples

    for i, (row_id, title, url, old_niche) in enumerate(headlines, start=1):
        # ── fetch article text (used for both scoring and classification) ───────
        article_text = fetch_article_text(url) if url else None
        score_text   = article_text or title

        # ── sentiment score ─────────────────────────────────────────────────────
        sc = score_finbert(score_text)

        # ── niche reclassification ──────────────────────────────────────────────
        new_niche, method = classify_niche(title, article_text or "")

        batch_buf.append((sc["compound"], sc["pos"], sc["neg"], sc["neu"], new_niche, row_id))

        # ── per-article progress line ───────────────────────────────────────────
        title_preview = title[:65] + "…" if len(title) > 65 else title
        niche_changed = old_niche != new_niche
        arrow = f"{old_niche} → {new_niche}" if niche_changed else f"{old_niche} (unchanged)"
        print(f"  [{i:4d}/{total}] {title_preview}")
        print(f"           niche: {arrow}  {method}")

        # ── flush batch ────────────────────────────────────────────────────────
        if len(batch_buf) >= BATCH_SIZE:
            update_scores(conn, batch_buf)
            updated += len(batch_buf)
            batch_buf.clear()
            pct = updated / total * 100
            print(f"\n  ── Progress: {updated}/{total} ({pct:.1f}%) updated ──\n")

    # ── flush remainder ────────────────────────────────────────────────────────
    if batch_buf:
        update_scores(conn, batch_buf)
        updated += len(batch_buf)
        batch_buf.clear()

    conn.close()
    print(f"\n✅  Done — {updated}/{total} headlines re-scored with FinBERT and niches reclassified.")


if __name__ == "__main__":
    main()
