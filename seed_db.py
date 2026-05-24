"""
seed_db.py
Seed the database with realistic demo headlines for all 15 niches.
Use this to demo the dashboard when NewsAPI isn't reachable.
"""

import sqlite3, random
from datetime import datetime, timezone, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import NICHES

DB_PATH = "sentiment.db"
analyzer = SentimentIntensityAnalyzer()

SAMPLE_HEADLINES = {
    "Bitcoin/Crypto": [
        "Bitcoin surges past $70,000 as institutional demand soars",
        "Ethereum ETF approval sparks crypto market rally",
        "Crypto exchange faces regulatory scrutiny amid fraud allegations",
        "Bitcoin mining energy concerns grow among ESG investors",
        "DeFi protocol hacked for $200M in latest crypto heist",
        "BlackRock Bitcoin ETF sees record inflows this week",
        "Crypto market tumbles as Fed signals prolonged high rates",
        "NFT market collapses 95% from peak, report finds",
        "El Salvador Bitcoin experiment shows mixed economic results",
        "Binance fined $4.3B in landmark crypto regulatory settlement",
    ],
    "Stock Market": [
        "S&P 500 hits new all-time high driven by tech surge",
        "Fed rate cut hopes lift equities to weekly gains",
        "Recession fears grip Wall Street as yield curve inverts",
        "Magnificent Seven stocks power index to record close",
        "Small-cap stocks rally as investors rotate from big tech",
        "Dow drops 600 points on hot inflation data",
        "Earnings season kicks off with better-than-expected results",
        "Market volatility spikes amid geopolitical uncertainty",
        "Warren Buffett trims Apple stake, raises cash reserves",
        "IPO market roars back with wave of tech listings",
    ],
    "Personal Finance": [
        "Credit card debt reaches record $1.3 trillion as rates bite",
        "High-yield savings accounts still offering 5% APY",
        "Americans cutting back on dining out to rebuild savings",
        "Student loan forgiveness program faces new legal challenges",
        "Financial advisors warn of retirement savings shortfall crisis",
        "Budgeting apps see 40% user surge as inflation persists",
        "Home affordability at worst level since 1980s, report shows",
        "Side hustles surge as workers seek financial cushion",
        "529 plan contributions hit record amid college cost fears",
        "Americans' emergency fund balances hit multi-year lows",
    ],
    "Electric Vehicles": [
        "Tesla cuts prices again as competition intensifies",
        "Ford EV division posts $1.3B quarterly loss",
        "GM accelerates EV rollout following strong Silverado demand",
        "EV adoption slows as charging infrastructure gaps persist",
        "Rivian secures massive fleet deal with Amazon for delivery vans",
        "China dominates global EV market with 60% share",
        "EV battery costs fall 40% in two years, analysts say",
        "Used EV prices crash 30%, raising residual value concerns",
        "BYD outsells Tesla globally for second straight quarter",
        "New tax credits boost EV sales among middle-income buyers",
    ],
    "Solar/Home Energy": [
        "Residential solar installations hit record in Q1",
        "Home battery storage demand soars amid grid outages",
        "Solar panel costs drop to historic lows, driving adoption",
        "Net metering cuts threaten rooftop solar economics",
        "Federal solar tax credit extension boosts installer stocks",
        "Community solar programs expand access to renters",
        "Solar company bankruptcies raise customer warranty concerns",
        "Heat pumps outsell gas furnaces for first time",
        "Grid-scale solar deployment surpasses coal capacity",
        "Permitting delays slow solar buildout despite demand",
    ],
    "Meal Kit Delivery": [
        "HelloFresh reports declining subscribers amid value concerns",
        "Blue Apron files for bankruptcy after subscriber collapse",
        "Meal kit industry pivots to ready-to-eat as cooking fatigue sets in",
        "Grocery delivery wars squeeze meal kit margins further",
        "Factor Meals sees growth as consumers seek convenience",
        "Meal kit waste concerns drive consumers to alternatives",
        "Dinnerly launches ultra-budget tier to win back price-sensitive customers",
        "Subscription fatigue hitting meal kit retention rates hard",
        "Pandemic-era meal kit boom officially over, data shows",
        "Martha Stewart partners with new meal kit startup",
    ],
    "Supplements": [
        "Creatine sales surge 80% as mainstream wellness interest grows",
        "FDA warns consumers about contaminated weight loss supplements",
        "Protein powder market hits $25B on fitness boom",
        "GLP-1 drugs compete with supplement industry for weight loss consumers",
        "Magnesium glycinate emerges as top trending sleep supplement",
        "Third-party testing gaps concern experts as supplement market grows",
        "Vitamin D deficiency awareness drives record supplement sales",
        "Collagen supplements lack clinical evidence, researchers warn",
        "Athletic Greens faces FTC scrutiny over health claims",
        "Pre-workout supplement market booms among Gen Z gym-goers",
    ],
    "AI Tools": [
        "OpenAI launches GPT-5 with multimodal reasoning capabilities",
        "AI productivity tools cut white-collar work hours by 30%, study finds",
        "Google Gemini Ultra rivals GPT-4 in new benchmarks",
        "AI copyright lawsuits mount as content creators push back",
        "Anthropic Claude gains enterprise market share over rivals",
        "AI hallucinations remain critical challenge for business adoption",
        "Microsoft Copilot adoption reaches 1M enterprise users",
        "AI coding assistants now write 40% of GitHub code",
        "EU AI Act enforcement begins, reshaping global AI compliance",
        "Startups raise record $50B in AI funding in 2024",
    ],
    "Smartphones": [
        "iPhone 16 pre-orders break Apple sales records",
        "Samsung Galaxy S25 Ultra debuts with advanced AI features",
        "Smartphone market shrinks for third consecutive year globally",
        "Apple Vision Pro sales disappoint amid high price concerns",
        "Foldable phone adoption growing but mainstream remains distant",
        "Google Pixel 9 wins camera benchmark crown from iPhone",
        "Qualcomm chip shortage delays Android flagship launches",
        "Chinese phone makers gain ground in Europe with lower prices",
        "Repair Right movement forces Apple to expand parts program",
        "5G adoption hits 50% of global smartphone users milestone",
    ],
    "Streaming Services": [
        "Netflix password sharing crackdown drives subscriber surge",
        "Disney+ raises prices again as streaming wars intensify",
        "Ad-supported tiers now generate more revenue than subscriptions",
        "Streaming fatigue drives record cancellation rates in Q3",
        "Amazon Prime Video ad insertion angers longtime subscribers",
        "Max surpasses 100M subscribers after Warner merger payoff",
        "Live sports streaming becomes key battleground for platforms",
        "Peacock grows on back of Olympics and NFL deals",
        "Streaming profitability finally materializing for major players",
        "Cord-cutting acceleration continues as cable loses 5M subs",
    ],
    "Remote Work": [
        "Amazon mandates 5-day return to office starting January",
        "Remote work productivity higher than in-office, Stanford study finds",
        "Cities offering incentives to attract remote workers",
        "RTO mandates linked to talent exodus at major firms",
        "Hybrid work becomes dominant model for knowledge workers",
        "Commercial real estate crisis deepens as offices stay empty",
        "Remote workers earn 20% more than office peers on average",
        "Zoom fatigue gives way to async-first work culture",
        "Dell, JPMorgan join wave of full RTO policy announcements",
        "Co-working space demand spikes in suburban markets",
    ],
    "Travel/Airlines": [
        "Summer airfare prices soar 25% above pre-pandemic levels",
        "Delta reports record quarterly profit on strong leisure demand",
        "Spirit Airlines bankruptcy leaves travelers scrambling",
        "TSA checkpoint volume hits 2024 record ahead of Thanksgiving",
        "Hotel room rates at all-time highs in major US cities",
        "Southwest eliminates open seating policy in major overhaul",
        "International travel demand surges to post-COVID records",
        "Flight cancellations spike amid ATC staffing shortages",
        "Budget airlines under pressure as fuel costs remain high",
        "Cruise industry books out 18 months in advance on demand surge",
    ],
    "Fitness": [
        "Peloton reports another quarterly loss as subscriber count drops",
        "Weight-loss drugs reshape gym and supplement industry outlook",
        "Planet Fitness membership reaches record 19M on low-cost appeal",
        "Pickleball mania drives $500M in court construction spending",
        "Home gym equipment sales collapse post-pandemic",
        "Zone 2 cardio trend drives treadmill sales to new highs",
        "CrossFit affiliate gyms down 20% from peak as fad fades",
        "Running shoe market booms with marathon participation records",
        "Fitness tracker market grows as health monitoring goes mainstream",
        "Corporate wellness programs return as firms fight burnout",
    ],
    "Real Estate": [
        "Existing home sales hit 30-year low as lock-in effect persists",
        "Mortgage rates above 7% for second consecutive year",
        "New home construction surges as builders target first-time buyers",
        "Home prices rise 5% year-over-year despite affordability crisis",
        "Commercial real estate values fall 20% from peak",
        "iBuyer market collapses as Opendoor cuts operations",
        "Luxury real estate market booms despite broader slowdown",
        "Home insurance crisis deepens in Florida and California",
        "Foreign investment in US real estate hits decade low",
        "First-time buyers fall to record low share of market",
    ],
    "Rent/Apartments": [
        "National average rent falls for first time in three years",
        "NYC rent reaches record $4,100 median for one-bedroom",
        "Multifamily construction boom finally eases rent inflation",
        "Eviction filings surge back above pre-pandemic levels",
        "Rent control expansion sparks landlord exodus in several cities",
        "Rent-to-income ratios at worst levels in history, study finds",
        "Build-to-rent communities surge amid homeownership barriers",
        "Landlord AI tools face scrutiny for algorithmic rent-fixing",
        "Short-term rental bans ease housing pressure in resort towns",
        "Section 8 voucher backlog swells as demand outpaces supply",
    ],
}


def seed():
    conn = sqlite3.connect(DB_PATH)
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

    SOURCES = ["Reuters", "Bloomberg", "WSJ", "CNBC", "AP News", "Financial Times", "Axios", "The Verge", "TechCrunch"]
    now = datetime.now(timezone.utc)
    total = 0

    for niche, headlines in SAMPLE_HEADLINES.items():
        rows = []
        for i, title in enumerate(headlines):
            sc = analyzer.polarity_scores(title)
            hours_ago = random.randint(1, 72)
            pub = (now - timedelta(hours=hours_ago)).isoformat()
            rows.append((
                niche, title,
                random.choice(SOURCES),
                f"https://example.com/article-{abs(hash(title)) % 99999}",
                pub,
                sc["compound"], sc["pos"], sc["neu"], sc["neg"],
                now.isoformat(),
            ))
        conn.executemany("""
            INSERT INTO headlines
              (niche, title, source, url, published, compound, pos, neu, neg, fetched)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, rows)
        conn.commit()
        total += len(rows)
        print(f"  {niche}: {len(rows)} headlines seeded.")

    conn.close()
    print(f"\n✅  Seeded {total} headlines across {len(SAMPLE_HEADLINES)} niches → {DB_PATH}")


if __name__ == "__main__":
    seed()
