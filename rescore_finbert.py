"""
rescore_finbert.py
Re-score every headline already stored in PostgreSQL with FinBERT.

Two-pass design to beat the 6-hour GitHub Actions timeout:
  Pass 1 — Parallel article text fetching via ThreadPoolExecutor(max_workers=16).
            ~5 min for 5,000+ URLs instead of ~30 min sequential.
  Pass 2 — Batched FinBERT scoring (batch_size=32) and batched BART niche
            classification (batch_size=8, keyword-first so most rows never
            hit BART at all).

Scores and niche labels are committed to the DB in chunks of 200 rows so that
a timeout at 80% still preserves 80% of the work (reruns are idempotent).

Usage:
    python3 rescore_finbert.py [--only-niche NICHE_NAME] [--limit N]

Environment:
    DATABASE_URL — PostgreSQL connection string (required; set by Railway
                   automatically, or export it locally before running).
"""

import argparse
import concurrent.futures
import os

import psycopg2

from fetch_news import classify_niche, fetch_article_text, score_finbert  # noqa: F401 – kept for compatibility

# ── connection config ──────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set")
    exit(1)

CHUNK_SIZE = 200   # rows committed per DB transaction in Pass 2


# ── helpers ────────────────────────────────────────────────────────────────────

def get_conn():
    """Return an open psycopg2 connection."""
    return psycopg2.connect(DATABASE_URL)


def fetch_all_headlines(conn, only_niche: str | None = None, limit: int | None = None):
    """Return headline rows as a list of (id, title, url, niche) tuples.

    Optional filters:
      only_niche — restrict to rows WHERE niche = only_niche
      limit      — append LIMIT N to the query
    """
    sql = "SELECT id, title, url, niche FROM headlines"
    params: list = []
    if only_niche:
        sql += " WHERE niche = %s"
        params.append(only_niche)
    sql += " ORDER BY id"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
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


def score_sentiment_consumer_batch(texts: list[str], batch_size: int = 8) -> list[dict]:
    """Batched consumer-framed sentiment via BART zero-shot.
    Returns list of {compound, pos, neg, neu} dicts in input order."""
    from fetch_news import SENTIMENT_LABELS, _get_classifier, analyzer
    classifier = _get_classifier()
    framed_texts = []
    for t in texts:
        snippet = (t or "")[:800].strip()
        if not snippet:
            framed_texts.append(None)
        else:
            framed_texts.append("An investor reading this news would react: " + snippet)

    results: list[dict] = [None] * len(texts)
    # Process non-empty inputs in batches
    queue = [(i, ft) for i, ft in enumerate(framed_texts) if ft is not None]
    for i in range(0, len(queue), batch_size):
        chunk = queue[i:i + batch_size]
        chunk_texts = [ft for _, ft in chunk]
        try:
            preds = classifier(chunk_texts, SENTIMENT_LABELS)
            if isinstance(preds, dict):
                preds = [preds]
            for (orig_idx, _), pred in zip(chunk, preds):
                labels = pred["labels"]
                scores = pred["scores"]
                top_label = labels[0]
                top_score = scores[0]
                second_label = labels[1]
                second_score = scores[1]

                # Margin rule: if neutral barely wins, flip to the second-place directional label
                if "Indifferently" in top_label and (top_score - second_score) < 0.10:
                    top_label = second_label
                    top_score = second_score

                if "Bullishly" in top_label:
                    results[orig_idx] = {"compound": top_score, "pos": top_score, "neg": 0.0, "neu": 0.0}
                elif "Bearishly" in top_label:
                    results[orig_idx] = {"compound": -top_score, "pos": 0.0, "neg": top_score, "neu": 0.0}
                else:
                    results[orig_idx] = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": top_score}
        except Exception as e:
            print(f"    [!] Consumer sentiment batch error — falling back to VADER for {len(chunk)} items: {e}")
            for orig_idx, _ in chunk:
                results[orig_idx] = analyzer.polarity_scores(texts[orig_idx] or "")
    # Fill in empty inputs as neutral
    for i, r in enumerate(results):
        if r is None:
            results[i] = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
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
        snippet = (article_text or title or "")[:800].strip()
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
    parser = argparse.ArgumentParser(description="Re-score headlines with FinBERT.")
    parser.add_argument(
        "--only-niche", metavar="NICHE_NAME",
        help="Restrict to headlines WHERE niche = NICHE_NAME",
    )
    parser.add_argument(
        "--limit", type=int, metavar="N",
        help="Process only the first N headlines (useful for smoke tests)",
    )
    args = parser.parse_args()

    print("Connecting to PostgreSQL…")
    conn = get_conn()

    print("Fetching all headlines…")
    headlines = fetch_all_headlines(conn, only_niche=args.only_niche, limit=args.limit)
    total = len(headlines)
    if args.only_niche:
        print(f"Found {total} headlines in niche '{args.only_niche}' to re-score.\n")
    else:
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

    # ── Pass 2: chunked scoring, classification, and incremental DB commits ────
    print("Pass 2: Batched consumer-framed BART sentiment scoring and niche classification "
          f"(committing every {CHUNK_SIZE} rows)…\n")

    chunks = [headlines[i:i + CHUNK_SIZE] for i in range(0, total, CHUNK_SIZE)]
    total_chunks = len(chunks)
    rows_done = 0

    for chunk_n, chunk in enumerate(chunks, start=1):
        # Build inputs for this chunk
        score_texts = [
            article_texts.get(row_id) or title
            for row_id, title, url, old_niche in chunk
        ]
        titles_and_texts = [
            (title, article_texts.get(row_id) or "")
            for row_id, title, url, old_niche in chunk
        ]

        # Score and classify
        scores = score_sentiment_consumer_batch(score_texts)
        niches = classify_niche_batch(titles_and_texts)

        # Build update tuples and commit immediately
        batch_buf: list[tuple] = [
            (sc["compound"], sc["pos"], sc["neg"], sc["neu"], new_niche, row_id)
            for (row_id, title, url, old_niche), sc, (new_niche, method)
            in zip(chunk, scores, niches)
        ]
        update_scores(conn, batch_buf)
        rows_done += len(batch_buf)
        pct = rows_done / total * 100
        print(f"  ✓ Committed chunk {chunk_n}/{total_chunks} "
              f"({rows_done}/{total} rows, {pct:.1f}%)")

    conn.close()
    print(f"\n✅  Done — {rows_done}/{total} headlines re-scored with consumer-framed BART and niches reclassified.")


if __name__ == "__main__":
    main()
