"""Sanity-check golden_dataset.jsonl (run in CI before evaluation)."""
import json, sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "golden_dataset.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
need = {"id", "question", "expected_answer", "answerable"}
errors = []
for r in rows:
    if need - r.keys():
        errors.append(f"{r.get('id', '?')}: missing {sorted(need - r.keys())}")
    if not isinstance(r.get("answerable"), bool):
        errors.append(f"{r.get('id', '?')}: 'answerable' must be true/false")
dupes = [k for k, v in Counter(r["id"] for r in rows).items() if v > 1]
if dupes:
    errors.append(f"duplicate ids: {dupes}")
print(f"{len(rows)} items · categories: {dict(Counter(r.get('category', '?') for r in rows))}")
if len(rows) < 100:
    print("WARNING: fewer than 100 items - grow the golden set (see scripts/generate_candidates.py)")
if errors:
    print("\n".join(errors)); sys.exit(1)
