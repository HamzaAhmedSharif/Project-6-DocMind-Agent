import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", encoding="utf-8-sig", override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LLM backend: any OpenAI-compatible endpoint.
#   ollama = model running locally on your own GPU (no quota, no cost)
#   hosted = a free-tier API that speaks the OpenAI protocol (Groq, Mistral, ...); set HOSTED_* in .env
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

if LLM_PROVIDER == "ollama":
    LLM_BASE_URL = "http://localhost:11434/v1"
    LLM_API_KEY = "ollama"  # Ollama ignores the key, but the client needs a non-empty value
    LLM_MODEL = os.getenv("OLLAMA_MODEL", "docmind-llama")
    LLM_RPM = 0  # 0 = no request pacing
    LLM_TPM = 0  # 0 = no token pacing
    LLM_EXTRA_BODY = {}
else:
    LLM_BASE_URL = os.getenv("HOSTED_BASE_URL")
    LLM_API_KEY = os.getenv("HOSTED_API_KEY")
    LLM_MODEL = os.getenv("HOSTED_MODEL")
    LLM_RPM = int(os.getenv("HOSTED_RPM", "5"))
    LLM_TPM = int(os.getenv("HOSTED_TPM", "0"))  # provider tokens-per-minute limit; 0 = no token pacing
    # Optional provider-specific request fields as JSON, e.g. {"reasoning_effort": "low"}
    LLM_EXTRA_BODY = json.loads(os.getenv("HOSTED_EXTRA_BODY", "") or "{}")
    if not (LLM_BASE_URL and LLM_API_KEY and LLM_MODEL):
        raise ValueError("LLM_PROVIDER=hosted needs HOSTED_BASE_URL, HOSTED_API_KEY and HOSTED_MODEL in .env")

SUPERVISOR_MODEL = LLM_MODEL  # routing + review
WRITER_MODEL = LLM_MODEL      # drafting + M2 baseline answers

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 25
TOP_K_RESULTS = 10

DATA_DIR = "data"
INDEX_PATH = "data/faiss_index"
