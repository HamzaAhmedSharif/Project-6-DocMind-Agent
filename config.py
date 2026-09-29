import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", encoding="utf-8-sig", override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

LLM_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 4

DATA_DIR = "data"
INDEX_PATH = "data/faiss_index"

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set. Copy .env.example to .env and add your key.")