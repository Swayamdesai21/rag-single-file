import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------
# LLM Configuration
# -------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq | openai

# OpenAI (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = "gpt-4o-mini"

# Groq (recommended)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_LLM_MODEL = "llama-3.1-8b-instant"

if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")

# -------------------------
# Embedding Configuration
# -------------------------
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")

# HuggingFace configurations
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# -------------------------
# Chunking Configuration
# -------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# -------------------------
# Qdrant Configuration
# -------------------------
COLLECTION_NAME = "single_file_rag"
QDRANT_URL = os.getenv("QDRANT_URL") # For remote Qdrant (e.g. Qdrant Cloud)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # For remote Qdrant

# -------------------------
# Retrieval Configuration
# -------------------------
TOP_K = 20

