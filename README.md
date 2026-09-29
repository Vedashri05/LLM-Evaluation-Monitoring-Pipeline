# LLM Eval CI/CD Pipeline

Unit tests for your LLM app. Every time a prompt, model, RAG knowledge base or pipeline code changes,
CI runs a golden dataset through the system, scores it, and **blocks the merge** if quality, latency or cost regress.

```
git push ─▶ GitHub Actions ─▶ run golden set through RAG ─▶ judge answers ─▶ metrics
                                                                             │
        PR comment + red ❌ (merge blocked) ◀── gate: SLA + baseline ◀───────┘
                                                                             │
                     main branch ─▶ results/history.jsonl ─▶ Streamlit dashboard
```

## Metrics
| Metric | How it's computed |
|---|---|
| Hallucination rate | % answers with faithfulness < cutoff, **plus** answers given to unanswerable questions |
| Faithfulness | share of claims in the answer supported by retrieved context |
| Answer relevancy / correctness | how well the answer addresses the question / matches the reference |
| Latency p50 / p95 | end-to-end (retrieval + generation) per query |
| Cost per query | token usage × price table in `config.yaml` |

Judges: `heuristic` (free token-overlap, for plumbing/smoke tests) or `anthropic` / `openai` (LLM-as-judge, use this for real numbers).

## Quick start (no API key needed)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/validate_dataset.py
python -m evals.run_eval --record --update-baseline   # first run: creates baseline.json + history
streamlit run dashboard/app.py
```
Default `provider: mock` is an offline extractive stand-in so you can see everything working.

## Use a real model
1. In `config.yaml` set `generation.provider` (`anthropic` | `openai`) and `model`; optionally `judge.provider`.
2. `export ANTHROPIC_API_KEY=...` (or `OPENAI_API_KEY`). In GitHub: Settings → Secrets → Actions.
3. Check `pricing:` numbers against the provider's current pricing page.
4. Use a stronger/different model for the judge than the one under test where possible.

## Grow the golden dataset to 100+
`golden_dataset.jsonl` ships with 18 examples. Draft more with
`python -m scripts.generate_candidates --per-chunk 6`, **review by hand**, then append.
Aim for a mix: factual, edge cases (negation, numbers), unanswerable, adversarial/prompt-injection.
Replace `knowledge_base/` and the prompt with your own product docs.

## Wire up the merge block
1. Push to GitHub (workflow: `.github/workflows/llm-eval.yml`).
2. Settings → Branches → add rule for `main` → **Require status checks** → select `evaluate`.
3. Commit `baseline.json` (from `--update-baseline`) so regression checks (p95 +20%, hallucination +2pp, relevancy −5pp) activate.
   Refresh it deliberately when you accept a new normal.

## Layout
```
app/          rag.py (system under test), llm.py (anthropic/openai/mock), textutil.py
evals/        run_eval.py (entrypoint) · judges.py · metrics.py · gates.py
dashboard/    app.py (Streamlit)
scripts/      validate_dataset.py · generate_candidates.py
prompts/      system_prompt.txt   knowledge_base/   golden_dataset.jsonl
config.yaml   thresholds, model, pricing        baseline.json   regression reference
```

## Notes / limits
- Latency in CI depends on runner + API load; keep `eval.workers: 1` for stable numbers and leave headroom in the SLA.
- LLM judges are noisy: fix `temperature: 0`, use a fixed judge model, and spot-check flagged samples.
- Judge failures fail closed (count as hallucination) so a broken judge can't let a bad change through.
