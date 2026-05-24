# Market Sentiment Analytics Platform

> Real-time NLP sentiment scoring across 15 consumer niches using VADER, NewsAPI, and an interactive dashboard.

## What it does

Pulls live news headlines for 15 consumer verticals, scores each headline using VADER sentiment analysis, stores results in SQLite, and surfaces actionable signals on an interactive dashboard.

**15 niches covered:**
Bitcoin/Crypto · Stock Market · Personal Finance · Electric Vehicles · Solar/Home Energy · Meal Kit Delivery · Supplements · AI Tools · Smartphones · Streaming Services · Remote Work · Travel/Airlines · Fitness · Real Estate · Rent/Apartments

## Stack

| Layer | Tech |
|---|---|
| Data ingestion | NewsAPI (free tier) |
| NLP scoring | VADER Sentiment (vaderSentiment) |
| Storage | SQLite via Python sqlite3 |
| Backend API | FastAPI (optional, via `api.py`) |
| Scheduling | `schedule` library (6-hour refresh) |
| Dashboard | React + Chart.js (interactive) |

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-tracker
cd sentiment-tracker
pip install vaderSentiment requests schedule
```

Add your NewsAPI key to `config.py`, then:

```bash
# Pull live headlines and score them
python3 fetch_news.py

# Or seed with demo data
python3 seed_db.py

# Auto-refresh every 6 hours
python3 scheduler.py
```

## Project structure

```
sentiment_tracker/
├── config.py        # API key + 15 niche keyword lists
├── fetch_news.py    # Ingestion: NewsAPI → VADER → SQLite
├── seed_db.py       # Demo data seeder (150 realistic headlines)
├── api.py           # Data access layer (queryable from dashboard)
├── scheduler.py     # 6-hour auto-refresh loop
├── sentiment.db     # SQLite database (auto-created)
└── README.md
```

## Sentiment scoring

Each headline is scored using VADER's `polarity_scores()`, producing:
- **compound** — normalized score from −1 (most negative) to +1 (most positive)
- **pos / neu / neg** — proportion of text in each category

Signals are aggregated per niche and surfaced as ranked sentiment indices.

## Resume description

> Market Sentiment Analytics Platform (Python / VADER / React)  
> Engineered a multi-niche sentiment pipeline targeting 15 consumer verticals using VADER NLP, SQLite, and a React/Chart.js dashboard. Built a resilient data ingestion layer with NewsAPI delivering structured sentiment scoring to a queryable backend across 15 consumer niches.

## License

MIT
