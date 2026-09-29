"""Provider-agnostic LLM client (anthropic | openai | mock)."""
from __future__ import annotations
import re
import time
from dataclasses import dataclass

from app.textutil import REFUSAL, content_tokens


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def cost_usd(pricing: dict, model: str, in_tok: int, out_tok: int) -> float:
    p = pricing.get(model)
    if not p:
        return 0.0
    return (in_tok * p["input"] + out_tok * p["output"]) / 1_000_000


class LLMClient:
    def __init__(self, provider: str, model: str, max_tokens: int = 400, temperature: float = 0.0):
        self.provider, self.model = provider, model
        self.max_tokens, self.temperature = max_tokens, temperature
        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
        elif provider == "openai":
            from openai import OpenAI
            self._client = OpenAI()                       # reads OPENAI_API_KEY
        elif provider == "mock":
            self._client = None
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ------------------------------------------------------------------
    def generate(self, system: str, user: str, retries: int = 3) -> LLMResult:
        last_err = None
        for attempt in range(retries):
            try:
                return self._generate_once(system, user)
            except Exception as e:                        # noqa: BLE001
                last_err = e
                time.sleep(1.5 ** attempt)
        raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")

    def _generate_once(self, system: str, user: str) -> LLMResult:
        start = time.perf_counter()
        if self.provider == "anthropic":
            r = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                system=system, messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            it, ot = r.usage.input_tokens, r.usage.output_tokens
        elif self.provider == "openai":
            r = self._client.chat.completions.create(
                model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            text = r.choices[0].message.content or ""
            it, ot = r.usage.prompt_tokens, r.usage.completion_tokens
        else:
            text = self._mock(user)
            it, ot = int(len((system + user).split()) * 1.3), int(len(text.split()) * 1.3)
        return LLMResult(text.strip(), it, ot, (time.perf_counter() - start) * 1000)

    # ------------------------------------------------------------------
    @staticmethod
    def _mock(user: str) -> str:
        """Offline stand-in: extractive answer, refuses when nothing matches well.
        Lets you run the whole pipeline (and CI) without an API key."""
        m = re.search(r"CONTEXT:\n(.*)\n\nQUESTION:\s*(.*)", user, re.S)
        if not m:
            return REFUSAL
        context, question = m.group(1), m.group(2).strip()
        q = content_tokens(question)
        lines = [l for l in context.splitlines() if not re.match(r"^\[\d+\]", l)]
        best, best_score = "", 0.0
        for sent in re.split(r"(?<=[.!?])\s+", " ".join(lines)):
            if q:
                score = len(q & content_tokens(sent)) / len(q)
                if score > best_score:
                    best, best_score = sent, score
        return best if best_score >= 0.6 else REFUSAL
