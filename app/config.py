import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_env_or_warn(key, default=None):
    val = os.getenv(key, default)
    if not val and not default:
        print(f"WARNING: Environment variable {key} is missing!", flush=True)
    return val

# -------------------------
# LLM Configuration
# -------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_LLM_MODEL = "llama-3.1-8b-instant"

# -------------------------
# Embedding Configuration
# -------------------------
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# -------------------------
# Qdrant Configuration
# -------------------------
COLLECTION_NAME = "single_file_rag"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# -------------------------
# App Configuration
# -------------------------
TOP_K = 20
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def check_config():
    errors = []
    if not GROQ_API_KEY: errors.append("GROQ_API_KEY is missing")
    if os.getenv("VERCEL") and not HUGGINGFACEHUB_API_TOKEN:
        errors.append("HUGGINGFACEHUB_API_TOKEN is missing (required for Vercel)")
    if QDRANT_URL and not QDRANT_API_KEY:
        errors.append("QDRANT_API_KEY is missing for remote Qdrant")
    return errors
