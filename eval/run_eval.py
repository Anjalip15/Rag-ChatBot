"""Small RAG evaluation: retrieval hit-rate + answer keyword accuracy + refusal on out-of-scope questions.

Usage (server NOT required; needs GEMINI_API_KEY):
    python -m eval.run_eval --doc docs/sample_document.txt
"""
import argparse
import json
from pathlib import Path

from app.rag_pipeline import RAGPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="docs/sample_document.txt")
    ap.add_argument("--eval", default="eval/eval_set.json")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    rag = RAGPipeline()
    rag.add_document(args.doc)
    cases = json.loads(Path(args.eval).read_text(encoding="utf-8"))

    hit = ans_ok = n_in = 0
    for c in cases:
        res = rag.answer(c["question"], top_k=args.k)
        text = res["answer"].lower()
        ok = all(k.lower() in text for k in c["must_contain"])
        kw = c.get("expected_source_keyword")
        if kw:
            n_in += 1
            hit += any(kw.lower() in s["text"].lower() for s in res["sources"])
            ans_ok += ok
        print(f"{'PASS' if ok else 'FAIL'} | {c['question']}\n      -> {res['answer'][:120]!r}")
        if not kw and not ok:
            print("      (hallucination risk: answered an out-of-scope question)")

    print("\n=== Results ===")
    print(f"Retrieval hit-rate@{args.k}: {hit}/{n_in}")
    print(f"Answer accuracy (in-scope):  {ans_ok}/{n_in}")


if __name__ == "__main__":
    main()
