"""Draft new golden QA pairs from your knowledge base with an LLM.
OUTPUT MUST BE HUMAN-REVIEWED before being appended to golden_dataset.jsonl.

  python -m scripts.generate_candidates --per-chunk 6 --provider anthropic --model claude-haiku-4-5-20251001
"""
import argparse, json, re
from pathlib import Path

import yaml

from app.llm import LLMClient
from app.rag import RAGPipeline

PROMPT = """From the passage below write {n} question/answer pairs a real user might ask.
Mix: simple facts, tricky edge cases (negations, numbers), and 1-2 questions the passage does NOT answer
(answerable=false, expected_answer="I don't know based on the provided documents.").
Return ONLY a JSON list of objects with keys: question, expected_answer, answerable (bool), category
(factual|edge|unanswerable).

PASSAGE:
{text}"""

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--per-chunk", type=int, default=6)
ap.add_argument("--provider", default="anthropic")
ap.add_argument("--model", default="claude-haiku-4-5-20251001")
ap.add_argument("--out", default="golden_candidates.jsonl")
a = ap.parse_args()

cfg = yaml.safe_load(Path(a.config).read_text())
chunks = RAGPipeline(cfg).chunks
llm = LLMClient(a.provider, a.model, 1500, 0.3)
rows = []
for i, c in enumerate(chunks):
    raw = llm.generate("You write evaluation datasets. Output JSON only.",
                       PROMPT.format(n=a.per_chunk, text=c["text"])).text
    try:
        for j, r in enumerate(json.loads(re.search(r"\[.*\]", raw, re.S).group(0))):
            rows.append({"id": f"c{i:02d}_{j}", "source": c["source"], **r})
    except Exception as e:      # noqa: BLE001
        print(f"chunk {i}: parse failed ({e})")
Path(a.out).write_text("\n".join(json.dumps(r) for r in rows))
print(f"Wrote {len(rows)} candidates -> {a.out}. Review, fix, then append to golden_dataset.jsonl")
