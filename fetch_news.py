"""
fetch_news.py
Pull headlines from NewsAPI + Reuters/AP RSS feeds, classify each article
into one of 16 niches using a hybrid approach:
  1. High-confidence keyword matching (no model needed — instant).
  2. facebook/bart-large-mnli zero-shot NLI fallback when no keyword matches.
Scores sentiment with local FinBERT (ProsusAI/finbert via transformers).
Falls back to VADER on FinBERT error; falls back to "Politics & Economy" on
classifier error.
Uses PostgreSQL when DATABASE_URL env var is set, otherwise falls back to SQLite.
Run:  python3 fetch_news.py
"""

import email.utils
import json
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

from anthropic import Anthropic

from config import ANTHROPIC_API_KEY, NEWS_API_KEY

_claude_client = Anthropic(api_key=ANTHROPIC_API_KEY, timeout=30.0, max_retries=2) if ANTHROPIC_API_KEY else None

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
# Descriptive keyword phrases fed to the zero-shot classifier — richer context
# gives the NLI model more signal than bare category names.
NICHE_LABELS: list[str] = [
    "oil prices, energy sector, OPEC, petroleum, natural gas",
    "housing market, real estate, mortgage, home prices, rent",
    "stock market, banking, Federal Reserve, inflation, Wall Street, bonds",
    "food prices, agriculture, farming, grocery, crops",
    "retail shopping, e-commerce, Amazon, consumer spending, Walmart",
    "social media, digital advertising, TikTok, Meta, Facebook",
    "airlines, travel, tourism, hotels, airfare",
    "video games, gaming industry, esports, Nintendo, PlayStation",
    "healthcare, pharmaceuticals, biotech, FDA, medical, vaccines",
    "semiconductors, chips, NVIDIA, Intel, TSMC, microchips",
    "cybersecurity, hacking, data breach, ransomware, cyber attack",
    "electric vehicles, Tesla, EV, battery, charging",
    "bitcoin, cryptocurrency, blockchain, ethereum, crypto",
    "space exploration, SpaceX, NASA, rockets, satellites",
    "artificial intelligence, machine learning, AI, ChatGPT, OpenAI",
    "politics, government policy, Congress, White House, elections, geopolitics",
]

# Maps each keyword label back to its human-readable canonical niche name.
NICHE_LABEL_MAP: dict[str, str] = {
    "oil prices, energy sector, OPEC, petroleum, natural gas": "Energy & Oil",
    "housing market, real estate, mortgage, home prices, rent": "Real Estate",
    "stock market, banking, Federal Reserve, inflation, Wall Street, bonds": "Banking & Finance",
    "food prices, agriculture, farming, grocery, crops": "Food & Agriculture",
    "retail shopping, e-commerce, Amazon, consumer spending, Walmart": "Retail & E-Commerce",
    "social media, digital advertising, TikTok, Meta, Facebook": "Social Media & AdTech",
    "airlines, travel, tourism, hotels, airfare": "Travel & Tourism",
    "video games, gaming industry, esports, Nintendo, PlayStation": "Gaming & Esports",
    "healthcare, pharmaceuticals, biotech, FDA, medical, vaccines": "Healthcare & Biotech",
    "semiconductors, chips, NVIDIA, Intel, TSMC, microchips": "Semiconductors",
    "cybersecurity, hacking, data breach, ransomware, cyber attack": "Cybersecurity",
    "electric vehicles, Tesla, EV, battery, charging": "Electric Vehicles",
    "bitcoin, cryptocurrency, blockchain, ethereum, crypto": "Bitcoin & Crypto",
    "space exploration, SpaceX, NASA, rockets, satellites": "Space Technology",
    "artificial intelligence, machine learning, AI, ChatGPT, OpenAI": "AI & Machine Learning",
    "politics, government policy, Congress, White House, elections, geopolitics": "Politics & Economy",
}

# ── high-confidence keyword signals for instant niche assignment ───────────────
# If ANY keyword phrase appears in (title + first 200 chars of article text),
# that niche wins immediately — the AI classifier is never called.
NICHE_KEYWORDS: dict[str, list[str]] = {
    "Energy & Oil": [
        "crude oil", "OPEC", "oil price", "Brent", "WTI", "petroleum",
        "natural gas price", "oil barrel", "Strait of Hormuz", "oil production",
        "gasoline price", "LNG", "oil refin",
    ],
    "Real Estate": [
        "housing market", "home price", "mortgage rate", "real estate",
        "home sale", "commercial real estate", "housing start", "rent price",
        "foreclosure", "home builder", "housing affordab",
    ],
    "Banking & Finance": [
        "Federal Reserve", "interest rate", "S&P 500", "Dow Jones", "Nasdaq",
        "bond yield", "JPMorgan", "Goldman Sachs", "Fed rate", "treasury yield",
        "stock market", "inflation rate", "bank earn",
        "gold price", "silver price", "precious metal",
    ],
    "Food & Agriculture": [
        "food price", "agriculture", "grocery", "farming", "wheat price",
        "corn future", "crop yield", "food supply", "USDA", "food inflation",
        "livestock", "soybean", "fertilizer",
    ],
    "Retail & E-Commerce": [
        "retail sale", "Amazon earn", "consumer spending", "e-commerce",
        "Walmart earn", "Target earn", "consumer confidence", "holiday sale",
        "online shopping", "supply chain retail",
        "auto parts", "AutoZone", "O'Reilly Auto",
    ],
    "Social Media & AdTech": [
        "Meta earn", "TikTok", "Instagram", "digital advertising",
        "Facebook stock", "ad spending", "Snapchat", "YouTube revenue",
        "social media stock", "ad revenue",
    ],
    "Travel & Tourism": [
        "airline earn", "airfare", "Delta Air", "United Airlines",
        "American Airlines", "hotel occupancy", "tourism spending",
        "cruise industry", "airport traffic", "flight price",
    ],
    "Gaming & Esports": [
        "video game sale", "Nintendo", "PlayStation", "Xbox", "gaming revenue",
        "esports", "Steam sale", "gaming stock", "mobile gaming", "Activision",
        "Electronic Arts", "Roblox",
    ],
    "Healthcare & Biotech": [
        "FDA approval", "FDA approved", "biotech stock", "pharmaceutical earn",
        "clinical trial", "Medicare fund", "Medicaid", "drug approval",
        "vaccine", "Pfizer", "Moderna", "UnitedHealth", "Eli Lilly",
        "hospital stock",
        "supplement", "gummies", "GABA", "tinnitus", "psychedelic",
        "brain health", "weight loss drug",
    ],
    "Semiconductors": [
        "semiconductor", "NVIDIA earn", "Intel earn", "chip shortage", "TSMC",
        "Qualcomm", "AMD earn", "microchip", "chip export", "chip stock",
        "wafer", "Micron", "Broadcom",
        "Marvell", "memory chip", "semiconductor stock",
    ],
    "Cybersecurity": [
        "data breach", "ransomware", "cybersecurity", "hacking", "cyber attack",
        "malware", "phishing", "CrowdStrike", "Palo Alto", "cyber threat",
        "zero-day",
    ],
    "Electric Vehicles": [
        "electric vehicle", "Tesla earn", "EV sale", "EV battery",
        "charging station", "BYD", "Rivian", "Lucid Motors", "EV market",
        "electric car", "EV stock",
    ],
    "Bitcoin & Crypto": [
        "bitcoin price", "bitcoin ETF", "ethereum price", "crypto market cap",
        "cryptocurrency exchange", "Coinbase revenue", "crypto regulation",
        "DeFi protocol", "stablecoin", "crypto trading", "blockchain network",
        "NFT market", "crypto wallet", "binance", "crypto fund",
    ],
    "Space Technology": [
        "SpaceX", "NASA", "rocket launch", "satellite deploy", "Starship",
        "Blue Origin", "space economy", "space stock", "orbital launch",
        "space exploration",
    ],
    "AI & Machine Learning": [
        "artificial intelligence earn", "ChatGPT", "OpenAI", "Anthropic",
        "generative AI", "machine learning stock", "LLM", "AI regulation",
        "NVIDIA AI", "AI investment", "AI chip", "foundation model",
        "AI executive order", "David Sacks",
    ],
    "Politics & Economy": [
        "Congress vote", "Senate bill", "White House policy", "President Trump",
        "tariff", "trade war", "GDP", "recession", "federal deficit",
        "government shutdown", "debt ceiling", "federal budget",
        "economic sanction", "geopolitical", "election result",
        "executive order", "Treasury Secretary",
        "Republican", "Democrat", "Senate", "House vote", "campaign",
        "primary election",
    ],
}

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
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_title_source ON headlines(title, source)"
    )
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
    """Return the FinBERT pipeline, loading it on the first call.

    Auto-selects device: MPS (Apple Silicon GPU) > CUDA (NVIDIA GPU) > CPU.
    Falls back to CPU on any device error.
    """
    global _finbert_pipeline
    if _finbert_pipeline is None:
        import torch
        # Device selection: MPS for Apple Silicon, CUDA for NVIDIA, else CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"  [finbert] Loading ProsusAI/finbert on {device.upper()} (first run only)…")
        try:
            _finbert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=device,
            )
        except Exception as e:
            print(f"  [finbert] {device.upper()} load failed — falling back to CPU: {e}")
            _finbert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device="cpu",
            )
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


# ── consumer-framed BART sentiment scoring ────────────────────────────────────

SENTIMENT_LABELS = [
    "Bullishly — stock prices and consumer financial outlook are likely to rise",
    "Bearishly — stock prices and consumer financial outlook are likely to fall",
    "Indifferently — this news has no meaningful impact on stocks or consumer finances",
]


def score_sentiment_consumer(text: str) -> dict:
    """Score consumer-framed sentiment using BART zero-shot.

    Returns a dict with keys: compound, pos, neg, neu — same shape as score_finbert.
    Maps:
      good news → compound=+score, pos=score, neg=0, neu=0
      bad news  → compound=-score, pos=0, neg=score, neu=0
      neutral   → compound=0, pos=0, neg=0, neu=score
    """
    try:
        classifier = _get_classifier()
        snippet = text[:800].strip() if text else ""
        if not snippet:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        framed = "An investor reading this news would react: " + snippet
        result = classifier(framed, SENTIMENT_LABELS)
        labels = result["labels"]
        scores = result["scores"]
        top_label = labels[0]
        top_score = scores[0]
        second_label = labels[1]
        second_score = scores[1]

        # Margin rule: if neutral barely wins, flip to the second-place directional label
        if "Indifferently" in top_label and (top_score - second_score) < 0.10:
            top_label = second_label
            top_score = second_score

        if "Bullishly" in top_label:
            return {"compound": top_score, "pos": top_score, "neg": 0.0, "neu": 0.0}
        elif "Bearishly" in top_label:
            return {"compound": -top_score, "pos": 0.0, "neg": top_score, "neu": 0.0}
        else:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": top_score}
    except Exception as e:
        print(f"    [!] Consumer sentiment error — falling back to VADER: {e}")
        return analyzer.polarity_scores(text)


# ── zero-shot niche classifier ─────────────────────────────────────────────────

def _get_classifier():
    """Return the zero-shot classification pipeline, loading it on first call.

    Auto-selects device: MPS (Apple Silicon GPU) > CUDA (NVIDIA GPU) > CPU.
    Falls back to CPU on any device error.
    """
    global _classifier_pipeline
    if _classifier_pipeline is None:
        import torch
        # Device selection: MPS for Apple Silicon, CUDA for NVIDIA, else CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"  [classifier] Loading facebook/bart-large-mnli on {device.upper()} (first run only)…")
        try:
            _classifier_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device,
            )
        except Exception as e:
            print(f"  [classifier] {device.upper()} load failed — falling back to CPU: {e}")
            _classifier_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device="cpu",
            )
    return _classifier_pipeline


def classify_niche(title: str, article_text: str) -> tuple[str, str]:
    """Classify an article into one of the 16 canonical niches using a hybrid approach.

    Strategy
    --------
    1. **Title keyword pass** — scan *title only* against NICHE_KEYWORDS
       (case-insensitive).  The first niche whose any keyword phrase appears in
       the title wins immediately; the AI model is never loaded.
       Returns ``(niche, "[keyword-title]")``.

    2. **Text keyword pass** — if no title match, scan the *first 200 characters
       of article_text* against NICHE_KEYWORDS (case-insensitive).  First match
       wins; the AI model is still not loaded.
       Returns ``(niche, "[keyword-text]")``.

    3. **AI fallback** — if still no match, feed the first 2000 characters of
       *article_text* (or *title* when article_text is empty) to the zero-shot
       facebook/bart-large-mnli classifier.  Uses hypothesis-style framing
       ("This news article is primarily about: …") for better NLI signal.
       If the top score is below 0.35 (weak guess), returns
       ``("Politics & Economy", "[AI-low-confidence]")`` instead of forcing a
       random niche.  The winning label is mapped back via NICHE_LABEL_MAP.
       Returns ``(niche, "[AI]")`` on confident classification.

    Falls back to ``("Politics & Economy", "[AI]")`` on any classifier exception or
    empty input.
    """
    # ── Step 1: title-only keyword matching ────────────────────────────────────
    title_lower = title.lower()
    for niche, keywords in NICHE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in title_lower:
                return niche, "[keyword-title]"

    # ── Step 2: article text (first 200 chars) keyword matching ────────────────
    text_snippet = (article_text or "")[:200].lower()
    for niche, keywords in NICHE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_snippet:
                return niche, "[keyword-text]"

    # ── Step 3: zero-shot AI fallback ──────────────────────────────────────────
    try:
        snippet = (article_text or title or "")[:800].strip()
        if not snippet:
            return "Politics & Economy", "[AI]"
        framed_snippet = "This news article is primarily about: " + snippet
        classifier = _get_classifier()
        result     = classifier(framed_snippet, NICHE_LABELS)
        if result["scores"][0] < 0.35:
            return "Politics & Economy", "[AI-low-confidence]"
        return NICHE_LABEL_MAP[result["labels"][0]], "[AI]"   # map label → niche name
    except Exception as e:
        print(f"    [!] Classifier error — defaulting to Politics & Economy: {e}")
        return "Politics & Economy", "[AI]"


# ── Claude Haiku combined niche + sentiment scoring ───────────────────────────

def score_and_classify_claude(title: str, article_text: str) -> tuple[str, dict, str]:
    """Classify niche and score investor-reaction sentiment in one Claude Haiku call.

    Returns (niche, sentiment_dict, method_string).
    Falls back to classify_niche() + score_sentiment_consumer() on any failure.
    """
    if _claude_client is None:
        niche, _ = classify_niche(title, article_text)
        return niche, score_sentiment_consumer(article_text or title), "[fallback-bart]"

    try:
        niche_names = list(NICHE_LABEL_MAP.values())
        snippet = (article_text or title or "")[:1500].strip()

        prompt = (
            f"Article title: {title}\n\n"
            f"Article text (first 1500 characters):\n{snippet}\n\n"
            "Respond with ONLY valid JSON — no markdown, no extra text:\n"
            '{"niche": "<niche>", "sentiment": "bullish|bearish|neutral",'
            ' "score": <float -1.0 to 1.0>, "rationale": "<one sentence>"}\n\n'
            f"Valid niche names (pick exactly one): {', '.join(niche_names)}\n\n"
            "Scoring rules:\n"
            "- Routine SEC filings (Form 144, 8-K, 13F), scheduled insider transactions,"
            " procedural/administrative notices → neutral (score ≈ 0)\n"
            "- Contract wins, earnings beats, raised guidance, capital investment,"
            " expansion, acquisitions (for the acquirer) → bullish\n"
            "- Lawsuits, bankruptcies, missed earnings, layoffs, regulatory crackdowns,"
            " war/conflict escalation, consumer confidence drops → bearish\n"
            "- Score magnitude reflects impact: massive contract = +0.8, minor positive = +0.3"
            " (inverse for bearish)\n"
            "- IMPORTANT — always judge sentiment from the perspective of an INVESTOR in"
            " the relevant sector or asset: will this likely push related stock/asset"
            " prices UP (bullish) or DOWN (bearish)? Examples of this frame: falling oil"
            " prices = bearish for the energy sector; rising interest rates = bearish for"
            " stocks and bonds; rising wages = can be bearish for corporate margins. Judge"
            " market impact, not consumer convenience or literal word tone."
        )

        response = _claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        if "{" in raw:
            raw = raw[raw.index("{") : raw.rindex("}") + 1]
        data = json.loads(raw)

        niche = data.get("niche", "Politics & Economy")
        if niche not in niche_names:
            niche = "Politics & Economy"

        sentiment = str(data.get("sentiment", "neutral")).lower()
        score = max(-1.0, min(1.0, float(data.get("score", 0.0))))

        if sentiment == "bullish":
            sc = {"compound": score, "pos": abs(score), "neg": 0.0, "neu": 0.0}
        elif sentiment == "bearish":
            sc = {"compound": score, "pos": 0.0, "neg": abs(score), "neu": 0.0}
        else:
            sc = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        return niche, sc, "[claude]"

    except Exception as e:
        print(f"    [!] Claude scoring error — falling back to BART: {e}")
        niche, _ = classify_niche(title, article_text)
        return niche, score_sentiment_consumer(article_text or title), "[fallback-bart]"


# ── article text fetcher ──────────────────────────────────────────────────────

def fetch_article_text(url: str) -> str | None:
    """Fetch full article body text for richer sentiment/classification scoring.

    Makes a browser-like GET request, extracts all <p> tag text via
    BeautifulSoup, and returns the first 2500 characters (enough context for
    BART classification and FinBERT scoring).  Returns None when:
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
        return text[:2500]
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

    # ── 4. Skip articles already in the database ──────────────────────────────
    cur = conn.cursor()
    cur.execute("SELECT title FROM headlines")
    existing_titles: set[str] = {row[0].lower() for row in cur.fetchall()}
    cur.close()

    new_articles = [a for a in unique if a["title"].lower() not in existing_titles]
    already_in_db = len(unique) - len(new_articles)
    print(
        f"  DB check: {len(new_articles)} new  |  {already_in_db} already in database — skipping those\n"
    )

    # ── 5. Classify niche, score sentiment, collect DB rows ───────────────────
    scorer = "Claude Haiku" if _claude_client else "BART (fallback — no API key)"
    print(f"Classifying niches + scoring {len(new_articles)} articles via {scorer}…\n")

    niche_counts: dict[str, int] = {}
    rows: list[tuple] = []

    for i, a in enumerate(new_articles, 1):
        article_text        = fetch_article_text(a["url"]) or ""
        niche, sc, method   = score_and_classify_claude(a["title"], article_text)

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
        print(f"  [{i:3d}/{len(new_articles)}] {title_preview}")
        print(f"           → {niche}  {method}  compound={sc['compound']:+.2f}")

        time.sleep(0.05)

    # ── 6. Bulk insert ─────────────────────────────────────────────────────────
    placeholder = "%s" if USE_POSTGRES else "?"
    if USE_POSTGRES:
        insert_sql = f"""
            INSERT INTO headlines
              (niche, title, source, url, published, compound, pos, neu, neg, fetched)
            VALUES ({', '.join([placeholder] * 10)})
            ON CONFLICT (title, source) DO NOTHING
        """
    else:
        insert_sql = f"""
            INSERT OR IGNORE INTO headlines
              (niche, title, source, url, published, compound, pos, neu, neg, fetched)
            VALUES ({', '.join([placeholder] * 10)})
        """
    cur = conn.cursor()
    cur.executemany(insert_sql, rows)
    conn.commit()
    cur.close()
    conn.close()

    # ── 7. Summary ─────────────────────────────────────────────────────────────
    db_label = DATABASE_URL.split("@")[-1] if USE_POSTGRES else DB_PATH
    print(f"\n✅  Done — {len(rows)} headlines saved to {db_label}")
    print("\n── Niche breakdown ──────────────────────────────────────────────────")
    for niche in NICHE_LABEL_MAP.values():
        count = niche_counts.get(niche, 0)
        if count:
            bar = "█" * min(count, 50)
            print(f"  {niche:<25s} {count:3d}  {bar}")
    print("─────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
