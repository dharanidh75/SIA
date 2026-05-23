import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
DATABASE_URL   = os.getenv("DATABASE_URL", "sqlite:///./sailp.db")
REDIS_URL      = os.getenv("REDIS_URL", "")          # optional for v1
LANGSMITH_KEY  = os.getenv("LANGSMITH_API_KEY", "")  # optional

GROQ_MODEL     = "llama-3.3-70b-versatile"

# News API (optional — graceful fallback if missing)
NEWS_API_KEY   = os.getenv("NEWS_API_KEY", "")