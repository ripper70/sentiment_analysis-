from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime
from config import NEWS_API_KEY, NICHES

api = NewsApiClient(api_key=NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()
engine = create_engine("sqlite:///sentiment.db")

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            niche TEXT,
            headline TEXT,
            score REAL,
            label TEXT,
            fetched_at TEXT
        )
    """))
    conn.commit()

def label(score):
    if score >= 0.05: return "positive"
    if score <= -0.05: return "negative"
    return "neutral"

rows = []
for niche, keywords in NICHES.items():
    for keyword in keywords:
        articles = api.get_everything(q=keyword, language="en", sort_by="publishedAt", page_size=20)
        for a in articles.get("articles", []):
            headline = a["title"]
            score = analyzer.polarity_scores(headline)["compound"]
            rows.append({
                "niche": niche,
                "headline": headline,
                "score": score,
                "label": label(score),
                "fetched_at": datetime.now().isoformat()
            })

df = pd.DataFrame(rows)
df.to_sql("sentiment", engine, if_exists="append", index=False)
print(f"Saved {len(df)} headlines to sentiment.db")
