"""Streamlit dashboard: is the LLM system getting better or worse over time?
Run:  streamlit run dashboard/app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RES = ROOT / CFG["results_dir"]

st.set_page_config(page_title="LLM Eval Dashboard", page_icon="📊", layout="wide")
st.title("📊 LLM Evaluation & Monitoring")

hist_path = RES / "history.jsonl"
if not hist_path.exists():
    st.info("No history yet. Run `python -m evals.run_eval --record` (CI does this on every merge to main).")
    st.stop()

rows = [json.loads(l) for l in hist_path.read_text().splitlines() if l.strip()]
df = pd.json_normalize(rows)
df.columns = [c.replace("metrics.", "") for c in df.columns]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["passed"] = df["gate.passed"]
df = df.sort_values("timestamp").reset_index(drop=True)
df["label"] = df["git_sha"] + " · " + df["timestamp"].dt.strftime("%m-%d %H:%M") + " #" + df.index.astype(str)

sel = st.sidebar.selectbox("Branch", ["all"] + sorted(df["branch"].unique()))
if sel != "all":
    df = df[df["branch"] == sel].reset_index(drop=True)
if df.empty:
    st.warning("No runs for this branch.")
    st.stop()

last, prev = df.iloc[-1], (df.iloc[-2] if len(df) > 1 else None)
T = CFG["thresholds"]

st.subheader("Latest run: " + ("✅ passed" if last["passed"] else "❌ blocked"))
st.caption(f"`{last['git_sha']}` on `{last['branch']}` · {last['provider']}/{last['model']} · "
           f"prompt `{last['prompt_hash']}` · KB `{last['kb_hash']}` · {int(last['n_samples'])} questions")


def kpi(col, label, key, fmt, inverse=False, help_=None):
    v = last[key]
    delta = None if prev is None else fmt.format(v - prev[key])
    col.metric(label, fmt.format(v), delta, delta_color="inverse" if inverse else "normal", help=help_)


c = st.columns(6)
kpi(c[0], "Hallucination", "hallucination_rate", "{:.1%}", True, f"limit {T['max_hallucination_rate']:.0%}")
kpi(c[1], "Faithfulness", "faithfulness", "{:.2f}", False, f"min {T['min_faithfulness']}")
kpi(c[2], "Relevancy", "relevancy", "{:.2f}", False, f"min {T['min_relevancy']}")
kpi(c[3], "Latency p50 (ms)", "latency_p50_ms", "{:.0f}", True)
kpi(c[4], "Latency p95 (ms)", "latency_p95_ms", "{:.0f}", True, f"SLA {T['max_p95_latency_ms']} ms")
kpi(c[5], "Cost / query", "cost_per_query_usd", "${:.5f}", True, f"limit ${T['max_cost_per_query_usd']}")

x = df.set_index("label")
a, b = st.columns(2)
a.markdown("**Hallucination rate** (lower is better)")
a.line_chart(x[["hallucination_rate"]])
b.markdown("**Faithfulness & relevancy** (higher is better)")
b.line_chart(x[["faithfulness", "relevancy"]])
a, b = st.columns(2)
a.markdown("**Latency p50 / p95 (ms)**")
a.line_chart(x[["latency_p50_ms", "latency_p95_ms"]])
b.markdown("**Cost per query (USD)**")
b.line_chart(x[["cost_per_query_usd"]])

st.subheader("Run history")
cols = ["timestamp", "git_sha", "branch", "model", "passed", "hallucination_rate", "faithfulness",
        "relevancy", "latency_p95_ms", "cost_per_query_usd"]
st.dataframe(df[cols].iloc[::-1], use_container_width=True, hide_index=True)

samples_path = RES / "samples_latest.jsonl"
if samples_path.exists():
    st.subheader("Latest local run: problem samples")
    s = pd.DataFrame([json.loads(l) for l in samples_path.read_text().splitlines() if l.strip()])
    bad = s[(s["hallucinated"]) | (s["relevancy"] < 0.5) | (s["error"].notna())]
    st.dataframe(bad[["id", "category", "question", "answer", "expected_answer", "faithfulness", "relevancy"]],
                 use_container_width=True, hide_index=True)
