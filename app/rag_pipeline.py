"""RAG pipeline: load -> split -> embed -> store -> retrieve -> answer with citations."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.vector_store import make_store

NOT_FOUND_MSG = "I couldn't find this in the uploaded documents."

PROMPT = ChatPromptTemplate.from_template(
    """You are a careful assistant that answers questions using ONLY the numbered context below.
Rules:
- If the answer is not in the context, reply exactly: "{not_found}"
- Do not use outside knowledge.
- Cite the sources you used like [1], [2].

Context:
{context}

Question: {question}

Answer:"""
)


class RAGPipeline:
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL, google_api_key=config.GEMINI_API_KEY
        )
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL, google_api_key=config.GEMINI_API_KEY, temperature=0
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        self.store = make_store(self.embeddings)

    # ---------- ingestion ----------
    @staticmethod
    def _load(file_path: str):
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(file_path).load()
        if ext == ".txt":
            return TextLoader(file_path, encoding="utf-8").load()
        raise ValueError(f"Unsupported file type: {ext}")

    def add_document(self, file_path: str) -> int:
        """Index a file. Re-uploading the same filename replaces the old chunks (no duplicates)."""
        name = Path(file_path).name
        chunks = self.splitter.split_documents(self._load(file_path))
        if not chunks:
            raise ValueError("No text could be extracted from this file (scanned PDF?).")
        for c in chunks:
            c.metadata["source"] = name
        self.store.delete_source(name)
        self.store.add_documents(chunks)
        return len(chunks)

    # ---------- querying ----------
    def answer(self, question: str, top_k: int = config.DEFAULT_TOP_K) -> dict:
        hits = self.store.search(question, top_k)
        context = "\n\n".join(f"[{i}] ({m.get('source')}) {t}" for i, (t, m, _) in enumerate(hits, 1))
        chain = PROMPT | self.llm | StrOutputParser()
        text = chain.invoke({"context": context, "question": question, "not_found": NOT_FOUND_MSG})
        return {
            "question": question,
            "answer": text.strip(),
            "sources": [
                {"id": i, "file": m.get("source"), "page": m.get("page"), "distance": round(d, 4), "text": t}
                for i, (t, m, d) in enumerate(hits, 1)
            ],
            "model_used": config.LLM_MODEL,
        }

    # ---------- housekeeping ----------
    def stats(self) -> dict:
        return self.store.stats()

    def is_ready(self) -> bool:
        return self.stats()["chunks"] > 0

    def reset(self) -> None:
        self.store.clear()
