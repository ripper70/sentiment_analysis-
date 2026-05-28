import os

# ── API keys — set these in your environment (never hardcode secrets) ──────────
# Railway: add them in your project's Variables tab.
# Local:   export them in your shell or use a .env file + python-dotenv.
#   NEWS_API_KEY        = <your NewsAPI key>
#   GUARDIAN_API_KEY    = <your Guardian API key>
#   HUGGINGFACE_API_KEY = <your HuggingFace API key>
#   DATABASE_URL        = <auto-set by Railway PostgreSQL plugin>
NEWS_API_KEY        = os.environ.get("NEWS_API_KEY", "")
GUARDIAN_API_KEY    = os.environ.get("GUARDIAN_API_KEY", "")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")


def _load_anthropic_key():
    # 1. Environment variable (GitHub Actions secret)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    # 2. Local gitignored file
    try:
        with open(os.path.join(os.path.dirname(__file__), ".anthropic_key")) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

ANTHROPIC_API_KEY = _load_anthropic_key()

NICHES = {
    "Energy & Oil":          ["oil price", "crude oil", "OPEC", "petroleum", "natural gas", "energy prices", "Strait of Hormuz", "Iran oil"],
    "Real Estate":           ["real estate", "housing market", "home prices", "mortgage rates", "housing prices", "home sales", "commercial real estate"],
    "Banking & Finance":     ["federal reserve", "interest rates", "Wall Street", "S&P 500", "inflation", "bond yield", "Fed rate", "Goldman", "JPMorgan"],
    "Food & Agriculture":    ["food prices", "agriculture", "grocery prices", "farming", "food inflation", "crop", "wheat", "corn prices"],
    "Retail & E-Commerce":   ["retail sales", "Amazon", "consumer spending", "e-commerce", "Walmart", "Target", "consumer confidence"],
    "Social Media & AdTech": ["social media", "TikTok", "Meta", "Instagram", "digital advertising", "Facebook", "X Twitter"],
    "Travel & Tourism":      ["airline", "airfare", "hotel prices", "tourism", "travel demand", "Delta airlines", "United airlines", "American airlines"],
    "Gaming & Esports":      ["video game", "Nintendo", "PlayStation", "Xbox", "Rockstar", "gaming industry", "esports", "Steam"],
    "Healthcare & Biotech":  ["FDA approved", "biotech", "pharmaceutical", "clinical trial", "Medicare", "Medicaid", "drug approval", "vaccine", "hospital costs", "health insurance", "Pfizer", "Moderna", "Johnson & Johnson", "Eli Lilly", "UnitedHealth"],
    "Semiconductors":        ["semiconductor", "NVIDIA", "Intel", "chip shortage", "microchip", "AMD", "TSMC", "Qualcomm"],
    "Cybersecurity":         ["cybersecurity", "data breach", "ransomware", "hacking", "cyber attack", "malware", "phishing"],
    "Electric Vehicles":     ["electric vehicle", "Tesla", "EV sales", "EV battery", "charging station", "BYD", "Rivian", "Lucid"],
    "Bitcoin & Crypto":      ["bitcoin", "cryptocurrency", "ethereum", "crypto market", "blockchain", "Coinbase", "crypto exchange"],
    "Space Technology":      ["SpaceX", "NASA", "rocket launch", "satellite", "space mission", "Starship", "Blue Origin"],
    "AI & Machine Learning": ["artificial intelligence", "ChatGPT", "machine learning", "OpenAI", "claude", "anthropic", "generative AI", "LLM"],
    "Politics & Economy":    ["Congress", "Senate", "White House", "Trump", "tariff", "trade war", "GDP", "recession", "debt ceiling", "federal budget", "sanctions", "geopolitical"],
}
