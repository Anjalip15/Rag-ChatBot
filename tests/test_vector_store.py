import hashlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.vector_store import SimpleVectorStore


class FakeEmb(Embeddings):
    def _v(self, t):
        v = [0.0] * 128
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1
        return v

    def embed_documents(self, ts):
        return [self._v(t) for t in ts]

    def embed_query(self, t):
        return self._v(t)


def docs(src):
    return [Document(page_content="python is the main language", metadata={"source": src}),
            Document(page_content="the internship lasts six months", metadata={"source": src}),
            Document(page_content="certificate given on completion", metadata={"source": src})]


def test_search_returns_most_similar(tmp_path):
    s = SimpleVectorStore(FakeEmb(), str(tmp_path))
    s.add_documents(docs("a.txt"))
    top = s.search("how many months does the internship last", 1)[0]
    assert "internship" in top[0]


def test_reupload_replaces_and_persists(tmp_path):
    s = SimpleVectorStore(FakeEmb(), str(tmp_path))
    s.add_documents(docs("a.txt"))
    s.delete_source("a.txt")
    s.add_documents(docs("a.txt"))
    assert s.stats()["chunks"] == 3
    s2 = SimpleVectorStore(FakeEmb(), str(tmp_path))  # reload from disk
    assert s2.stats() == {"documents": ["a.txt"], "chunks": 3}


def test_clear_and_empty_search(tmp_path):
    s = SimpleVectorStore(FakeEmb(), str(tmp_path))
    s.add_documents(docs("a.txt"))
    s.clear()
    assert s.stats()["chunks"] == 0 and s.search("anything", 3) == []
