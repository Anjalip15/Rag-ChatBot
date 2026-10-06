"""Central configuration. All values can be overridden with environment variables / .env."""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Model names change often - check https://ai.google.dev/gemini-api/docs/models and override in .env
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "simple")  # "simple" (NumPy, default) or "chroma"
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_docs")

# Sizes are in CHARACTERS (RecursiveCharacterTextSplitter default), not tokens.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
