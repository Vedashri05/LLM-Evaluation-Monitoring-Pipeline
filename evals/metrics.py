"""Aggregate per-sample records into pipeline-level metrics."""
from __future__ import annotations
import numpy as np


def aggregate(records: list[dict]) -> dict:
    n = len(records)
    ok = [r for r in records if not r.get("error")]
    lat = [r["latency_ms"] for r in ok] or [0.0]
    mean = lambda k: float(np.mean([r[k] for r in ok])) if ok else 0.0   # noqa: E731
    return {
        "n_samples": n,
        "error_rate": round((n - len(ok)) / n, 4) if n else 0.0,
        "hallucination_rate": round(sum(r["hallucinated"] for r in ok) / len(ok), 4) if ok else 1.0,
        "faithfulness": round(mean("faithfulness"), 4),
        "relevancy": round(mean("relevancy"), 4),
        "correctness": round(mean("correctness"), 4),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
        "cost_per_query_usd": round(mean("cost_usd"), 6),
        "total_cost_usd": round(float(sum(r["cost_usd"] for r in ok)), 6),
        "avg_input_tokens": round(mean("input_tokens"), 1),
        "avg_output_tokens": round(mean("output_tokens"), 1),
    }
