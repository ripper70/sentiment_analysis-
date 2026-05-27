"""
rescore_finbert.py
Re-score every headline already stored in PostgreSQL with FinBERT.

Two-pass design to beat the 6-hour GitHub Actions timeout:
  Pass 1 — Parallel article text fetching via ThreadPoolExecutor(max_workers=16).
            ~5 min for 5,000+ URLs instead of ~30 min sequential.
  Pass 2 — Batched FinBERT scoring (batch_size=32) and batched BART niche
            classification (batch_size=8, keyword-first so most rows never
            hit BART at all).

Scores and niche labels are written back to the headlines table in chunks of 50.

Usage:
    python3 rescore_finbert.py

Environment:
    DATABASE_URL — PostgreSQL connection string (required; set by Railway
                   automatically, or export it locally before running).
"""

import concurrent.futures
import os

import psycopg2

from fetch_news import classify_niche, fetch_article_text, score_finbert  # noqa: F401 – kept for compatibility

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


# ── batched helpers ────────────────────────────────────────────────────────────

def score_finbert_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    """Batched version of score_finbert. Returns list of {compound, pos, neg, neu} dicts."""
    from fetch_news import _get_finbert, analyzer
    finbert = _get_finbert()
    framed = ["How does this news impact consumer investments and financial markets? " + t for t in texts]
    results = []
    for i in range(0, len(framed), batch_size):
        chunk = framed[i:i + batch_size]
        try:
            preds = finbert(chunk, truncation=True, max_length=256)
            for p in preds:
                label = p["label"].lower()
                s = p["score"]
                if label == "positive":
                    results.append({"compound": s, "pos": s, "neg": 0.0, "neu": 0.0})
                elif label == "negative":
                    results.append({"compound": -s, "pos": 0.0, "neg": s, "neu": 0.0})
                else:
                    results.append({"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": s})
        except Exception as e:
            print(f"    [!] FinBERT batch error — falling back to VADER for {len(chunk)} items: {e}")
            for t in chunk:
                results.append(analyzer.polarity_scores(t))
    return results


def classify_niche_batch(titles_and_texts: list[tuple[str, str]], batch_size: int = 8) -> list[tuple[str, str]]:
    """Batched version of classify_niche. Input: list of (title, article_text).
    Output: list of (niche, method). Runs keyword passes first (free), then
    batches only the leftovers through BART."""
    from fetch_news import NICHE_KEYWORDS, NICHE_LABELS, NICHE_LABEL_MAP, _get_classifier

    results: list[tuple[str, str] | None] = [None] * len(titles_and_texts)
    bart_queue: list[tuple[int, str]] = []  # (original_index, framed_snippet)

    # Step 1 + 2: keyword passes (no AI)
    for idx, (title, article_text) in enumerate(titles_and_texts):
        title_lower = title.lower()
        matched = False
        for niche, keywords in NICHE_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in title_lower:
                    results[idx] = (niche, "[keyword-title]")
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue
        text_snippet = (article_text or "")[:200].lower()
        for niche, keywords in NICHE_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_snippet:
                    results[idx] = (niche, "[keyword-text]")
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue
        # Queue for BART
        snippet = (article_text or title or "")[:2000].strip()
        if not snippet:
            results[idx] = ("Politics & Economy", "[AI-empty]")
        else:
            bart_queue.append((idx, "This news article is primarily about: " + snippet))

    # Step 3: batched BART (only on leftovers)
    if bart_queue:
        classifier = _get_classifier()
        for i in range(0, len(bart_queue), batch_size):
            chunk = bart_queue[i:i + batch_size]
            chunk_texts = [s for _, s in chunk]
            try:
                preds = classifier(chunk_texts, NICHE_LABELS)
                # When given a list, classifier returns a list of result dicts
                if isinstance(preds, dict):
                    preds = [preds]
                for (orig_idx, _), pred in zip(chunk, preds):
                    top_label = pred["labels"][0]
                    top_score = pred["scores"][0]
                    if top_score < 0.35:
                        results[orig_idx] = ("Politics & Economy", "[AI-low-confidence]")
                    else:
                        results[orig_idx] = (NICHE_LABEL_MAP[top_label], "[AI]")
            except Exception as e:
                print(f"    [!] BART batch error — defaulting to Politics & Economy for {len(chunk)} items: {e}")
                for orig_idx, _ in chunk:
                    results[orig_idx] = ("Politics & Economy", "[AI-error]")

    return results  # type: ignore


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Connecting to PostgreSQL…")
    conn = get_conn()

    print("Fetching all headlines…")
    headlines = fetch_all_headlines(conn)
    total = len(headlines)
    print(f"Found {total} headlines to re-score.\n")

    # ── Pass 1: parallel article text fetching ─────────────────────────────────
    print(f"Pass 1: Fetching article text for {total} URLs in parallel (16 workers)…")
    article_texts: dict[int, str | None] = {}
    fetch_count = 0

    def fetch_one(row):
        row_id, title, url, old_niche = row
        return row_id, fetch_article_text(url) if url else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(fetch_one, row): row for row in headlines}
        for future in concurrent.futures.as_completed(futures):
            row_id, text = future.result()
            article_texts[row_id] = text
            fetch_count += 1
            if fetch_count % 200 == 0:
                print(f"  … fetched {fetch_count}/{total} articles")

    fetched_ok = sum(1 for v in article_texts.values() if v)
    print(f"  ✓ Retrieved body text for {fetched_ok} / {total} URLs\n")

    # ── Pass 2: batched scoring and classification ─────────────────────────────
    print("Pass 2: Batched FinBERT scoring and BART classification…")

    score_texts = [
        article_texts.get(row_id) or title
        for row_id, title, url, old_niche in headlines
    ]
    titles_and_texts = [
        (title, article_texts.get(row_id) or "")
        for row_id, title, url, old_niche in headlines
    ]

    print(f"  Scoring {total} articles with FinBERT (batch_size=32)…")
    scores = score_finbert_batch(score_texts)

    print(f"  Classifying {total} articles into niches (batch_size=8)…")
    niches = classify_niche_batch(titles_and_texts)

    # ── Build batch buffer and bulk-update ────────────────────────────────────
    print("\nWriting results to database…")
    batch_buf: list[tuple] = []
    updated = 0

    for i, (row, sc, (new_niche, method)) in enumerate(zip(headlines, scores, niches), start=1):
        row_id, title, url, old_niche = row
        batch_buf.append((sc["compound"], sc["pos"], sc["neg"], sc["neu"], new_niche, row_id))

        # Cheap per-article progress: print every 100th article
        if i % 100 == 0:
            niche_changed = old_niche != new_niche
            title_preview = title[:65] + "…" if len(title) > 65 else title
            arrow = f"{old_niche} → {new_niche}" if niche_changed else f"{old_niche} (unchanged)"
            print(f"  [{i:4d}/{total}] {title_preview}")
            print(f"           niche: {arrow}  {method}")

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
