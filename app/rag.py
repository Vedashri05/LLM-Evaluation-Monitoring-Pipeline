"""Minimal RAG pipeline (TF-IDF retrieval + LLM). This is the 'system under test'."""
from __future__ import annotations
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.llm import LLMClient, cost_usd


class RAGPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        rc = cfg["rag"]
        self.top_k = rc["top_k"]
        self.system_prompt = Path(cfg["prompt_path"]).read_text().strip()
        self.chunks = self._load_chunks(Path(rc["kb_dir"]), rc["chunk_words"])
        if not self.chunks:
            raise RuntimeError(f"No documents found in {rc['kb_dir']}")
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])
        g = cfg["generation"]
        self.model = g["model"]
        self.llm = LLMClient(g["provider"], g["model"], g["max_tokens"], g["temperature"])
        self.pricing = cfg.get("pricing", {})

    @staticmethod
    def _load_chunks(kb_dir: Path, max_words: int) -> list[dict]:
        chunks = []
        for f in sorted(list(kb_dir.glob("*.md")) + list(kb_dir.glob("*.txt"))):
            buf, n = [], 0
            for para in [p.strip() for p in f.read_text().split("\n\n") if p.strip()]:
                w = len(para.split())
                if buf and n + w > max_words:
                    chunks.append({"source": f.name, "text": " ".join(buf)})
                    buf, n = [], 0
                buf.append(para)
                n += w
            if buf:
                chunks.append({"source": f.name, "text": " ".join(buf)})
        return chunks

    def retrieve(self, question: str) -> list[dict]:
        scores = linear_kernel(self.vec.transform([question]), self.matrix).ravel()
        top = scores.argsort()[::-1][: self.top_k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top if scores[i] > 0]

    @staticmethod
    def build_user_prompt(question: str, contexts: list[dict]) -> str:
        ctx = "\n\n".join(f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(contexts, 1))
        return f"CONTEXT:\n{ctx}\n\nQUESTION: {question}"

    def answer(self, question: str) -> dict:
        t0 = time.perf_counter()
        contexts = self.retrieve(question)
        res = self.llm.generate(self.system_prompt, self.build_user_prompt(question, contexts))
        return {
            "answer": res.text,
            "contexts": contexts,
            "latency_ms": (time.perf_counter() - t0) * 1000,   # retrieval + generation
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "cost_usd": cost_usd(self.pricing, self.model, res.input_tokens, res.output_tokens),
        }
