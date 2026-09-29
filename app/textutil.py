"""Small text helpers shared by the pipeline and the heuristic judge."""
import re

REFUSAL = "I don't know based on the provided documents."

STOP = set("""a an and are as at be by can do does did for from has have how i in is it
many much of on or that the there this to what when where which who why with you your my
me we our their his her its was were will would should could""".split())

_REFUSAL_RE = re.compile(
    r"(i don'?t know|i do not know|cannot find|can'?t find|not (mentioned|provided|covered|specified|"
    r"contained)|no information|unable to (find|answer))", re.I)


def stem(tok: str) -> str:
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok


def content_tokens(text: str) -> set:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {stem(t) for t in toks if t not in STOP and len(t) > 1}


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))
