"""Scoring: faithfulness, relevancy, correctness, hallucination.

judge.provider = heuristic  -> free, deterministic token-overlap scoring (good for smoke tests / CI plumbing)
judge.provider = anthropic|openai -> LLM-as-judge (use for real quality numbers)
"""
from __future__ import annotations
import json
import re

from app.llm import LLMClient
from app.textutil import content_tokens, is_refusal

JUDGE_SYSTEM = "You are a strict, impartial evaluator of a retrieval-augmented QA system. Output JSON only."

JUDGE_TEMPLATE = """Evaluate the ANSWER.

QUESTION: {question}

CONTEXT (the only allowed source of truth):
{context}

REFERENCE ANSWER: {expected}

ANSWER: {answer}

Return ONLY a JSON object:
{{"faithfulness": <0-1, fraction of claims in ANSWER supported by CONTEXT>,
  "relevancy": <0-1, how directly ANSWER addresses QUESTION>,
  "correctness": <0-1, agreement of ANSWER with REFERENCE ANSWER>}}"""


class Judge:
    def __init__(self, cfg: dict):
        j = cfg["judge"]
        self.provider = j["provider"]
        self.cutoff = j.get("faithfulness_cutoff", 0.7)
        self.llm = None if self.provider == "heuristic" else LLMClient(self.provider, j["model"], 300, 0.0)

    # ------------------------------------------------------------------
    def score(self, sample: dict, answer: str, contexts: list[dict]) -> dict:
        answerable = sample["answerable"]
        refused = is_refusal(answer)

        if refused:   # no claims made -> trivially faithful; right only if it *should* refuse
            good = 0.0 if answerable else 1.0
            s = {"faithfulness": 1.0, "relevancy": good, "correctness": good}
        elif not answerable:   # answered something it should not have known
            s = {"faithfulness": 0.0, "relevancy": 0.0, "correctness": 0.0}
        elif self.llm is None:
            s = self._heuristic(sample, answer, contexts)
        else:
            s = self._llm_judge(sample, answer, contexts)

        hallucinated = (not refused) and ((not answerable) or s["faithfulness"] < self.cutoff)
        return {**s, "refused": refused, "hallucinated": hallucinated}

    # ------------------------------------------------------------------
    @staticmethod
    def _heuristic(sample, answer, contexts):
        ans = content_tokens(answer)
        ctx = content_tokens(" ".join(c["text"] for c in contexts))
        exp = content_tokens(sample["expected_answer"])
        faith = len(ans & ctx) / len(ans) if ans else 0.0
        cover = len(exp & ans) / len(exp) if exp else 0.0
        return {"faithfulness": round(faith, 3), "relevancy": round(cover, 3), "correctness": round(cover, 3)}

    def _llm_judge(self, sample, answer, contexts):
        ctx = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(contexts, 1)) or "(none)"
        prompt = JUDGE_TEMPLATE.format(question=sample["question"], context=ctx,
                                       expected=sample["expected_answer"], answer=answer)
        for _ in range(2):
            try:
                raw = self.llm.generate(JUDGE_SYSTEM, prompt).text
                obj = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
                return {k: min(1.0, max(0.0, float(obj[k]))) for k in ("faithfulness", "relevancy", "correctness")}
            except Exception:      # noqa: BLE001
                continue
        # Judge failed twice -> fail closed (counts as hallucination) so bad judging can't sneak a merge through
        return {"faithfulness": 0.0, "relevancy": 0.0, "correctness": 0.0}
