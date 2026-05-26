NEWS_API_KEY       = "08b3d831bf3c4da7a58936bb55cc0b52"
GUARDIAN_API_KEY   = "004ae94b-8fdf-4616-93d7-3ba02c1b3206"
HUGGINGFACE_API_KEY = ""

# ── Railway environment variables ──────────────────────────────────────────────
# Set the following secrets in your Railway project's Variables tab:
#   NEWS_API_KEY        = <your NewsAPI key>
#   GUARDIAN_API_KEY    = <your Guardian API key>
#   HUGGINGFACE_API_KEY = <your HuggingFace API key>
#   DATABASE_URL        = <auto-set by Railway PostgreSQL plugin>

NICHES = {
    "Energy & Oil":          ["oil price", "crude oil", "OPEC", "petroleum", "natural gas", "energy prices", "Strait of Hormuz", "Iran oil"],
    "Real Estate":           ["real estate", "housing market", "home prices", "mortgage rates", "housing prices", "home sales", "commercial real estate"],
    "Banking & Finance":     ["federal reserve", "interest rates", "Wall Street", "S&P 500", "inflation", "bond yield", "Fed rate", "Goldman", "JPMorgan"],
    "Food & Agriculture":    ["food prices", "agriculture", "grocery prices", "farming", "food inflation", "crop", "wheat", "corn prices"],
    "Retail & E-Commerce":   ["retail sales", "Amazon", "consumer spending", "e-commerce", "Walmart", "Target", "consumer confidence"],
    "Social Media & AdTech": ["social media", "TikTok", "Meta", "Instagram", "digital advertising", "Facebook", "X Twitter"],
    "Travel & Tourism":      ["airline", "airfare", "hotel prices", "tourism", "travel demand", "Delta airlines", "United airlines", "American airlines"],
    "Gaming & Esports":      ["video game", "Nintendo", "PlayStation", "Xbox", "Rockstar", "gaming industry", "esports", "Steam"],
    "Healthcare & Biotech":  ["healthcare", "biotech", "FDA approval", "pharmaceutical", "drug trial", "clinical trial", "Medicare", "Medicaid"],
    "Semiconductors":        ["semiconductor", "NVIDIA", "Intel", "chip shortage", "microchip", "AMD", "TSMC", "Qualcomm"],
    "Cybersecurity":         ["cybersecurity", "data breach", "ransomware", "hacking", "cyber attack", "malware", "phishing"],
    "Electric Vehicles":     ["electric vehicle", "Tesla", "EV sales", "EV battery", "charging station", "BYD", "Rivian", "Lucid"],
    "Bitcoin & Crypto":      ["bitcoin", "cryptocurrency", "ethereum", "crypto market", "blockchain", "Coinbase", "crypto exchange"],
    "Space Technology":      ["SpaceX", "NASA", "rocket launch", "satellite", "space mission", "Starship", "Blue Origin"],
    "AI & Machine Learning": ["artificial intelligence", "ChatGPT", "machine learning", "OpenAI", "claude", "anthropic", "generative AI", "LLM"],
    "Politics & Economy":    ["Congress", "Senate", "White House", "Trump", "tariff", "trade war", "GDP", "recession", "debt ceiling", "federal budget", "sanctions", "geopolitical"],
}
