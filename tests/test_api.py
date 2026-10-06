"""Offline API tests using a fake pipeline (no API key or network needed)."""
import io

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_pipeline


class FakePipeline:
    def __init__(self):
        self.docs = {}

    def add_document(self, path):
        self.docs[path] = 3
        return 3

    def is_ready(self):
        return bool(self.docs)

    def stats(self):
        return {"documents": list(self.docs), "chunks": 3 * len(self.docs)}

    def answer(self, q, top_k=4):
        return {"question": q, "answer": "fake", "model_used": "fake",
                "sources": [{"id": 1, "file": "a.txt", "page": None, "distance": 0.1, "text": "hello"}]}

    def reset(self):
        self.docs.clear()


@pytest.fixture()
def client():
    fake = FakePipeline()
    app.dependency_overrides[get_pipeline] = lambda: fake
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_ask_before_upload_returns_400(client):
    assert client.post("/ask", json={"question": "hello there"}).status_code == 400


def test_upload_and_ask(client):
    r = client.post("/upload", files={"file": ("a.txt", io.BytesIO(b"hello world"), "text/plain")})
    assert r.status_code == 200 and r.json()["chunks_created"] == 3
    r = client.post("/ask", json={"question": "what is this?"})
    assert r.status_code == 200 and r.json()["sources"][0]["id"] == 1


def test_rejects_bad_extension(client):
    r = client.post("/upload", files={"file": ("x.exe", io.BytesIO(b"x"), "application/octet-stream")})
    assert r.status_code == 400


def test_path_traversal_filename_is_sanitised(client):
    r = client.post("/upload", files={"file": ("../../evil.txt", io.BytesIO(b"x"), "text/plain")})
    assert r.status_code == 200 and r.json()["filename"] == "evil.txt"
