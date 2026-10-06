"""Vector stores. Default = SimpleVectorStore (pure NumPy, no native/DB dependencies, works everywhere).
Optional = ChromaStore (set VECTOR_BACKEND=chroma in .env, needs `pip install langchain-chroma chromadb`)."""
import json
import os
import threading
import uuid
from pathlib import Path

import numpy as np

from app import config


class SimpleVectorStore:
    """Stores chunk text + metadata in JSON and embeddings in a .npy file.
    Retrieval = cosine similarity computed with NumPy (fine for thousands of chunks)."""

    def __init__(self, embeddings, directory: str):
        self.embeddings = embeddings
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "chunks.json"
        self.vec_path = self.dir / "embeddings.npy"
        self.lock = threading.Lock()
        self.records: list[dict] = []
        self.matrix = np.zeros((0, 1), dtype=np.float32)
        self._load()

    # ---- persistence ----
    def _load(self):
        if self.meta_path.exists() and self.vec_path.exists():
            self.records = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.matrix = np.load(self.vec_path)
            if len(self.records) != len(self.matrix):  # inconsistent files -> start clean
                self.records, self.matrix = [], np.zeros((0, 1), dtype=np.float32)

    def _save(self):
        tmp_json = self.meta_path.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps(self.records, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_json, self.meta_path)
        tmp_npy = self.dir / "embeddings.tmp.npy"
        np.save(tmp_npy, self.matrix)
        os.replace(tmp_npy, self.vec_path)

    # ---- API shared with ChromaStore ----
    def add_documents(self, docs):
        texts = [d.page_content for d in docs]
        vecs = np.array(self.embeddings.embed_documents(texts), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12  # normalise -> dot product = cosine
        with self.lock:
            self.records += [{"id": uuid.uuid4().hex, "text": d.page_content, "metadata": d.metadata} for d in docs]
            self.matrix = vecs if len(self.matrix) == 0 else np.vstack([self.matrix, vecs])
            self._save()

    def delete_source(self, name: str):
        with self.lock:
            keep = [i for i, r in enumerate(self.records) if r["metadata"].get("source") != name]
            if len(keep) == len(self.records):
                return
            self.records = [self.records[i] for i in keep]
            self.matrix = self.matrix[keep] if keep else np.zeros((0, 1), dtype=np.float32)
            self._save()

    def search(self, question: str, k: int):
        """Returns [(text, metadata, cosine_distance)] - lower distance = more similar."""
        with self.lock:
            if not self.records:
                return []
            q = np.array(self.embeddings.embed_query(question), dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-12
            sims = self.matrix @ q
            top = np.argsort(-sims)[:k]
            return [(self.records[i]["text"], self.records[i]["metadata"], float(1 - sims[i])) for i in top]

    def stats(self):
        files = sorted({r["metadata"].get("source") for r in self.records})
        return {"documents": files, "chunks": len(self.records)}

    def clear(self):
        with self.lock:
            self.records, self.matrix = [], np.zeros((0, 1), dtype=np.float32)
            self._save()


class ChromaStore:
    """Optional ChromaDB backend with the same interface."""

    def __init__(self, embeddings, directory: str):
        from langchain_chroma import Chroma
        self._Chroma, self.embeddings, self.directory = Chroma, embeddings, directory
        self.vs = self._new()

    def _new(self):
        return self._Chroma(collection_name="docs", persist_directory=self.directory,
                            embedding_function=self.embeddings, collection_metadata={"hnsw:space": "cosine"})

    def add_documents(self, docs):
        self.vs.add_documents(docs)

    def delete_source(self, name):
        found = self.vs.get(where={"source": name})
        if found["ids"]:
            self.vs.delete(ids=found["ids"])

    def search(self, question, k):
        return [(d.page_content, d.metadata, float(s)) for d, s in self.vs.similarity_search_with_score(question, k=k)]

    def stats(self):
        data = self.vs.get(include=["metadatas"])
        return {"documents": sorted({m.get("source") for m in data["metadatas"] if m}), "chunks": len(data["ids"])}

    def clear(self):
        self.vs.delete_collection()
        self.vs = self._new()


def make_store(embeddings):
    if config.VECTOR_BACKEND == "chroma":
        return ChromaStore(embeddings, config.CHROMA_DB_DIR)
    return SimpleVectorStore(embeddings, config.VECTOR_DB_DIR)
