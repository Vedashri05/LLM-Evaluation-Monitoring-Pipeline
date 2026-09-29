"""Run the golden dataset through the pipeline, score it, apply the merge gate.

Usage:
  python -m evals.run_eval                       # evaluate + gate (exit 1 on failure)
  python -m evals.run_eval --record              # also append to results/history.jsonl (dashboard)
  python -m evals.run_eval --update-baseline     # save this run as baseline.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.rag import RAGPipeline
from evals.gates import evaluate_gates
from evals.judges import Judge
from evals.metrics import aggregate


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def fingerprint(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(Path(p).name.encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:10]


def git(*args, default="local"):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip() or default
    except Exception:      # noqa: BLE001
        return default


def evaluate_sample(sample, pipe: RAGPipeline, judge: Judge) -> dict:
    base = {"id": sample["id"], "category": sample.get("category", ""), "question": sample["question"],
            "expected_answer": sample["expected_answer"], "answerable": sample["answerable"]}
    try:
        out = pipe.answer(sample["question"])
        sc = judge.score(sample, out["answer"], out["contexts"])
        return {**base, "answer": out["answer"], "sources": [c["source"] for c in out["contexts"]],
                **{k: out[k] for k in ("latency_ms", "input_tokens", "output_tokens", "cost_usd")}, **sc, "error": None}
    except Exception as e:      # noqa: BLE001
        return {**base, "answer": "", "sources": [], "latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0,
                "cost_usd": 0.0, "faithfulness": 0.0, "relevancy": 0.0, "correctness": 0.0,
                "refused": False, "hallucinated": False, "error": str(e)}


def render_markdown(run: dict, samples: list[dict]) -> str:
    g, m = run["gate"], run["metrics"]
    icon = "✅ PASSED" if g["passed"] else "❌ FAILED – merge blocked"
    lines = [f"## LLM Eval Gate: {icon}", "",
             f"`{run['git_sha']}` on `{run['branch']}` · {run['provider']}/{run['model']} · "
             f"{m['n_samples']} questions · prompt `{run['prompt_hash']}` · KB `{run['kb_hash']}`", "",
             "| Check | Value | Limit | Result |", "|---|---|---|---|"]
    for c in g["checks"]:
        lines.append(f"| {c['name']} ({c['kind']}) | {c['value']} | {c['op']} {c['limit']} | {'✅' if c['ok'] else '❌'} |")
    lines += ["", f"**Latency** p50 {m['latency_p50_ms']} ms · p95 {m['latency_p95_ms']} ms  "
                  f"**Cost/query** ${m['cost_per_query_usd']:.5f}  **Faithfulness** {m['faithfulness']}  **Relevancy** {m['relevancy']}"]
    if not g["baseline_used"]:
        lines.append("\n_No baseline.json found – regression checks skipped._")
    bad = [s for s in samples if s["hallucinated"] or s["error"] or s["relevancy"] < 0.5][:8]
    if bad:
        lines += ["", "<details><summary>Problem samples</summary>", "", "| id | question | answer | issue |", "|---|---|---|---|"]
        for s in bad:
            issue = "error" if s["error"] else "hallucination" if s["hallucinated"] else "low relevancy"
            lines.append(f"| {s['id']} | {s['question'][:60]} | {(s['answer'] or s['error'] or '')[:80]} | {issue} |")
        lines += ["", "</details>"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=0, help="only run first N samples (debug)")
    ap.add_argument("--record", action="store_true", help="append run to results/history.jsonl")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dataset = load_jsonl(cfg["dataset_path"])
    if args.limit:
        dataset = dataset[: args.limit]
    if len(dataset) < 100:
        print(f"⚠️  Golden dataset has {len(dataset)} items (target: 100+).", file=sys.stderr)

    pipe, judge = RAGPipeline(cfg), Judge(cfg)
    print(f"Running {len(dataset)} questions with {cfg['generation']['provider']}/{cfg['generation']['model']} ...")
    with ThreadPoolExecutor(max_workers=cfg["eval"]["workers"]) as ex:
        samples = list(ex.map(lambda s: evaluate_sample(s, pipe, judge), dataset))

    metrics = aggregate(samples)
    baseline_path = Path(cfg["baseline_path"])
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    gate = evaluate_gates(metrics, cfg, baseline)

    kb_files = [p for p in Path(cfg["rag"]["kb_dir"]).glob("*") if p.is_file()]
    run = {
        "run_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": os.environ.get("GITHUB_SHA", "")[:7] or git("rev-parse", "--short", "HEAD"),
        "branch": os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME") or git("rev-parse", "--abbrev-ref", "HEAD"),
        "provider": cfg["generation"]["provider"], "model": cfg["generation"]["model"],
        "judge": cfg["judge"]["provider"],
        "prompt_hash": fingerprint([cfg["prompt_path"]]),
        "kb_hash": fingerprint(kb_files),
        "dataset_hash": fingerprint([cfg["dataset_path"]]),
        "metrics": metrics, "gate": {"passed": gate["passed"], "checks": gate["checks"],
                                     "baseline_used": gate["baseline_used"]},
    }

    out = Path(cfg["results_dir"]); out.mkdir(exist_ok=True)
    (out / "samples_latest.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
    (out / "report.json").write_text(json.dumps(run, indent=2))
    md = render_markdown(run, samples)
    (out / "report.md").write_text(md, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a").write(md + "\n")

    if args.record:
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(run) + "\n")
    if args.update_baseline:
        baseline_path.write_text(json.dumps({"metrics": metrics, "git_sha": run["git_sha"],
                                             "timestamp": run["timestamp"]}, indent=2))
        print(f"Baseline updated -> {baseline_path}")

    print("\n" + md)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
