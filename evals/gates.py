"""Merge gate: absolute SLA thresholds + regression vs baseline."""
from __future__ import annotations


def evaluate_gates(m: dict, cfg: dict, baseline: dict | None) -> dict:
    t, rg = cfg["thresholds"], cfg["regression"]
    checks = []

    def add(name, value, op, limit, kind="threshold"):
        ok = value <= limit if op == "<=" else value >= limit
        checks.append({"name": name, "kind": kind, "value": round(value, 4), "op": op,
                       "limit": round(limit, 4), "ok": bool(ok)})

    add("hallucination_rate", m["hallucination_rate"], "<=", t["max_hallucination_rate"])
    add("faithfulness", m["faithfulness"], ">=", t["min_faithfulness"])
    add("relevancy", m["relevancy"], ">=", t["min_relevancy"])
    add("latency_p95_ms", m["latency_p95_ms"], "<=", t["max_p95_latency_ms"])
    add("cost_per_query_usd", m["cost_per_query_usd"], "<=", t["max_cost_per_query_usd"])
    add("error_rate", m["error_rate"], "<=", t["max_error_rate"])

    if baseline:
        b = baseline["metrics"]
        add("p95_latency_vs_baseline", m["latency_p95_ms"], "<=",
            b["latency_p95_ms"] * (1 + rg["max_p95_latency_increase_pct"] / 100) + 1.0, "regression")   # +1ms noise floor
        add("hallucination_vs_baseline", m["hallucination_rate"], "<=",
            b["hallucination_rate"] + rg["max_hallucination_rate_increase"], "regression")
        add("relevancy_vs_baseline", m["relevancy"], ">=",
            b["relevancy"] - rg["max_relevancy_drop"], "regression")

    failures = [c for c in checks if not c["ok"]]
    return {"passed": not failures, "checks": checks, "failures": failures, "baseline_used": bool(baseline)}
