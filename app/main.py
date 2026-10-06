"""RAG Chatbot API - FastAPI + LangChain + ChromaDB + Gemini."""
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config

app = FastAPI(
    title="RAG Chatbot API",
    description="Upload PDF/TXT files and ask questions. Answers are grounded in your documents and cite sources.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path(config.UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"


@lru_cache(maxsize=1)
def get_pipeline():
    """Created lazily so the app (and /health) still start if the API key is missing."""
    from app.rag_pipeline import RAGPipeline
    try:
        return RAGPipeline()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------- schemas ----------
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=config.DEFAULT_TOP_K, ge=1, le=10)


class Source(BaseModel):
    id: int
    file: Optional[str]
    page: Optional[int] = None
    distance: float
    text: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    model_used: str


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int


# ---------- routes ----------
@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["General"])
async def health():
    return {"status": "healthy"}


@app.get("/status", tags=["General"])
def status(rag=Depends(get_pipeline)):
    s = rag.stats()
    return {**s, "ready_to_answer": s["chunks"] > 0, "model": config.LLM_MODEL}


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
def upload(file: UploadFile = File(...), rag=Depends(get_pipeline)):
    name = Path(file.filename or "").name  # strips any path -> blocks path traversal
    if Path(name).suffix.lower() not in {".pdf", ".txt"}:
        raise HTTPException(400, "Only .pdf and .txt files are supported.")

    path = UPLOAD_DIR / name
    size = 0
    with open(path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > config.MAX_UPLOAD_MB * 1024 * 1024:
                f.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large (max {config.MAX_UPLOAD_MB} MB).")
            f.write(chunk)
    try:
        n = rag.add_document(str(path))
    except Exception as e:
        raise HTTPException(500, f"Failed to process document: {e}")
    return UploadResponse(message=f"'{name}' added to the knowledge base.", filename=name, chunks_created=n)


@app.post("/ask", response_model=AnswerResponse, tags=["Chat"])
def ask(req: QuestionRequest, rag=Depends(get_pipeline)):
    if not rag.is_ready():
        raise HTTPException(400, "No documents loaded yet. Upload one with POST /upload first.")
    try:
        return AnswerResponse(**rag.answer(req.question, top_k=req.top_k))
    except Exception as e:
        raise HTTPException(500, f"Failed to generate answer: {e}")


@app.delete("/reset", tags=["Documents"])
def reset(rag=Depends(get_pipeline)):
    rag.reset()
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    return {"message": "Knowledge base cleared."}
