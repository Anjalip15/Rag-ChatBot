# RAG Chatbot (FastAPI + LangChain + ChromaDB + Gemini)

Upload PDF/TXT files and ask questions. Answers are generated **only from retrieved document chunks**, with numbered source citations, and the bot says it can't find the answer when the context doesn't contain it.

> RAG reduces hallucination but does not eliminate it, which is why the project includes an evaluation script (see below).

## Features
- Document ingestion (PDF/TXT) -> chunking -> Gemini embeddings -> a persistent vector store (cosine similarity)
- Citation-grounded answers + "not found" fallback for out-of-scope questions
- Sources returned with file name, page number and similarity distance
- Re-uploading a file replaces its old chunks (no duplicates)
- Built-in chat UI at `/`, Swagger docs at `/docs`
- Upload safety: extension check, size limit, sanitised filenames
- Offline unit tests (`pytest`) and a retrieval/answer evaluation script

## Architecture
```
Upload -> load (PyPDFLoader/TextLoader) -> split (800 chars, 100 overlap)
       -> embed (Gemini embedding) -> vector store (NumPy cosine similarity, saved to ./vector_db)
Question -> embed -> top-k cosine search -> numbered context -> Gemini -> answer + sources
```

## Setup
```bash
python -m venv venv && venv\Scripts\activate      # Windows (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env                                # then put your GEMINI_API_KEY in .env
uvicorn app.main:app --reload
```
Open http://localhost:8000 (chat UI) or http://localhost:8000/docs.

**Never commit `.env` or hard-code API keys.** Model names are configurable in `.env` (`LLM_MODEL`, `EMBEDDING_MODEL`) because Google retires models regularly.

## API
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Chat UI |
| GET | `/health` | Health check |
| GET | `/status` | Loaded documents and chunk count |
| POST | `/upload` | Upload .pdf/.txt |
| POST | `/ask` | `{"question": "...", "top_k": 4}` -> answer + sources |
| DELETE | `/reset` | Clear knowledge base |

## Evaluation
```bash
python -m eval.run_eval --doc docs/sample_document.txt
```
Reports retrieval hit-rate@k, answer accuracy on in-scope questions, and whether the bot correctly refuses an out-of-scope question. Add your own cases in `eval/eval_set.json`.
*(Record your actual results here after running it, e.g. "Hit-rate@4: 5/5, accuracy: 5/5".)*

## Tests
```bash
pytest
```

## Deploy (Render)
`render.yaml` is included. Add `GEMINI_API_KEY` in the Render dashboard. Note: the free tier has an ephemeral disk, so the vector DB resets on redeploy/restart; re-upload documents or attach a persistent disk.

## Vector store backends
The default store is a small NumPy implementation (cosine similarity, saved to `./vector_db`) with no native dependencies, so it installs everywhere. For ChromaDB, run `pip install langchain-chroma chromadb` and set `VECTOR_BACKEND=chroma` in `.env`.

## Possible next steps
Hybrid search (BM25 + vectors), a reranker, streaming responses, chat history, Docker.
