#!/usr/bin/env python3
"""
Metrics for the Bangladesh Labour Act QA evaluation.

Provides:
  - Automatic overlap metrics: sentence BLEU (sacrebleu) and ROUGE-L.
  - Citation metrics: validity of cited sections against the Act, and
    precision/recall/F1 against gold sections (reuses the section-extraction
    approach from validate_and_improve_dataset.py).
  - Grounding: fraction of the answer's content n-grams found verbatim in the
    Act text (a strict lexical faithfulness proxy).
  - Refusal detection: heuristic phrase matching.
  - Optional LLM-as-judge (local model, different family) for faithfulness,
    completeness, and harmfulness — clearly an automatic proxy, not human eval.

This module is import-only (no side effects) so evaluate.py can reuse it.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import sacrebleu
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
ACT_PDF = ROOT / "data/input/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf"
_ACT_CACHE = ROOT / "data/final/act_normalized.txt"

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
SECTION_RE = re.compile(r"[Ss]ection[s]?\s+(\d+)")

# Declining to answer and appending a disclaimer are different behaviours, and
# conflating them corrupts the metric in the direction that matters most: a model
# that answers an out-of-scope income-tax question in full and closes with
# "consult a tax professional" would otherwise be scored as having correctly
# refused. Measured on the base model's hold-out answers, 9 of 14 flagged
# refusals matched only the "consult a professional" phrase and every one of the
# 14 was a substantive answer of more than 60 words.
DECLINE_PATTERNS = [
    r"\boutside\b.{0,30}\bscope\b",
    r"\bcannot\s+(?:help|assist|answer|provide|verify|confirm|determine)\b",
    r"\bcan'?t\s+(?:help|assist|answer|provide|verify|confirm|determine)\b",
    r"\b(?:am|'m)\s+(?:not\s+able|unable)\s+to\s+"
    r"(?:help|assist|answer|provide|verify|confirm|determine|find)\b",
    r"\bunable\s+to\s+(?:help|assist|answer|provide|verify|confirm|determine|find)\b",
    r"\bbeyond\s+(?:my|the)\s+(?:scope|expertise|knowledge)\b",
    r"\bonly\s+(?:provide|help|speak|assist|answer).{0,40}\bLabour\s+Act\b",
    r"\bnot\s+covered\s+by\s+the\s+Bangladesh\s+Labour\s+Act\b",
    r"\bdo(?:es)?\s+not\s+fall\s+(?:with)?in\b.{0,30}\bscope\b",
    r"\bI\s+don'?t\s+have\s+(?:information|access|the\s+ability)\b",
    # Forms observed in the out-of-scope generations that the first pass missed.
    r"\b(?:won'?t|will\s+not)\s+be\s+able\s+to\b",
    r"\bnot\s+equipped\s+to\b",
    r"\bnot\s+capable\s+of\b",
    r"\bnot\s+aware\s+of\s+any\b",
    # Hedged non-answers: "I'm not sure about the specific VAT rate", "not sure
    # if I can provide a weather forecast". Both decline the request.
    r"\bnot\s+sure\s+(?:about|if|whether)\b",
]
DECLINE_RE = re.compile("|".join(DECLINE_PATTERNS), re.IGNORECASE)

# A referral on its own is a disclaimer, not a refusal - unless it is
# essentially the whole answer.
DISCLAIMER_PATTERNS = [
    r"\bconsult\b.{0,40}\b(?:professional|specialist|authority|expert|lawyer|advocate)\b",
    r"\bseek\s+(?:legal|professional)\s+advice\b",
]
DISCLAIMER_RE = re.compile("|".join(DISCLAIMER_PATTERNS), re.IGNORECASE)

# Minimum words remaining, once the referral itself is removed, for a reply to
# count as having actually answered. Measuring the remainder rather than the
# whole reply matters: a short but real answer that closes with "consult a
# professional" is an answer, and judging by total length alone would score it as
# a refusal.
_SUBSTANTIVE_WORDS = 25
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Kept for backwards compatibility with anything importing the old name.
REFUSAL_PATTERNS = DECLINE_PATTERNS
REFUSAL_RE = DECLINE_RE


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    text = re.sub(r"[^\w\s]", " ", text)
    return text.lower().strip()


def extract_sections(text: str):
    return sorted({int(s) for s in SECTION_RE.findall(text or "")})


@lru_cache(maxsize=1)
def act_normalized() -> str:
    """Normalized text of the Act, cached to disk.

    pdfminer takes minutes on this document and several scripts need the text,
    so the normalized form is extracted once and reused.
    """
    if _ACT_CACHE.exists():
        return _ACT_CACHE.read_text(encoding="utf-8")
    from pdfminer.high_level import extract_text
    text = normalize(extract_text(str(ACT_PDF)))
    _ACT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _ACT_CACHE.write_text(text, encoding="utf-8")
    return text


_ACT_RAW_CACHE = ROOT / "data/final/act_raw.txt"

# Section bodies in the Act are headed "117. Annual leave with wages.- (1) ..."
SECTION_HEADING_RE = re.compile(r"(?m)^[ \t]*(\d{1,3})\.[ \t]+([A-Z][^\n]{0,120})")


@lru_cache(maxsize=1)
def act_raw() -> str:
    """Raw (un-normalized) text of the Act, cached to disk.

    Section structure survives only in the raw text: normalization strips the
    punctuation that marks a heading.
    """
    if _ACT_RAW_CACHE.exists():
        return _ACT_RAW_CACHE.read_text(encoding="utf-8")
    from pdfminer.high_level import extract_text
    text = extract_text(str(ACT_PDF))
    _ACT_RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _ACT_RAW_CACHE.write_text(text, encoding="utf-8")
    return text


@lru_cache(maxsize=1)
def act_sections() -> dict:
    """Map section number -> (title, body), parsed from the headings.

    Where a number appears more than once - the table of contents repeats every
    heading - the longest span wins, which is always the real body.
    """
    raw = act_raw()
    matches = list(SECTION_HEADING_RE.finditer(raw))
    best = {}
    for i, m in enumerate(matches):
        num = int(m.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[m.start():end]
        if num not in best or len(body) > len(best[num][1]):
            best[num] = (m.group(2).strip(), body)
    return best


@lru_cache(maxsize=1)
def valid_sections() -> frozenset:
    """Section numbers that actually exist in the Act (hallucination check).

    Both sources are needed. Parsing headings finds the sections the Act
    *defines*; scanning for "section N" finds the ones it *cross-references*.
    Using only the cross-references (as an earlier version did) recovered 96 of
    354 sections, so most real citations were being counted as hallucinated.
    """
    act = act_normalized()
    return frozenset(set(act_sections())
                     | set(extract_sections(act))
                     | {int(s) for s in re.findall(r"section\s+(\d+)", act)})


# ---- overlap metrics --------------------------------------------------------

def bleu(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    return sacrebleu.sentence_bleu(pred, [ref]).score  # 0..100


def rouge_l(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    return _ROUGE.score(ref, pred)["rougeL"].fmeasure * 100  # 0..100


# ---- citation metrics -------------------------------------------------------

def citation_metrics(pred: str, gold_sections, valid_sections):
    """Return validity + precision/recall/F1 of cited sections.

    validity: of sections cited by the model, fraction that exist in the Act.
    precision/recall/F1: only meaningful when gold_sections is non-empty.
    """
    cited = set(extract_sections(pred))
    gold = set(gold_sections or [])
    valid = set(valid_sections or [])

    out = {
        "n_cited": len(cited),
        "cited_valid_frac": (len(cited & valid) / len(cited)) if cited else None,
        "has_citation": len(cited) > 0,
    }
    if gold:
        tp = len(cited & gold)
        out["cite_precision"] = tp / len(cited) if cited else 0.0
        out["cite_recall"] = tp / len(gold)
        p, r = out["cite_precision"], out["cite_recall"]
        out["cite_f1"] = (2 * p * r / (p + r)) if (p + r) else 0.0
    else:
        out["cite_precision"] = out["cite_recall"] = out["cite_f1"] = None
    return out


# ---- lexical grounding (strict faithfulness proxy) --------------------------

def grounding_score(pred: str, act_normalized: str, n: int = 5) -> float:
    """Fraction of the answer's n-grams that appear verbatim in the Act text.

    Stricter than any-single-4gram matching: it measures how much of the answer
    is lexically supported by the statute. Refusals/short answers return None.
    """
    words = normalize(pred).split()
    if len(words) < n:
        return None
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return None
    hits = sum(1 for g in grams if g in act_normalized)
    return hits / len(grams)


# ---- refusal ----------------------------------------------------------------

def is_refusal(pred: str) -> bool:
    """True when the answer declines to answer, rather than merely disclaiming.

    An explicit decline counts outright. A referral to a professional counts only
    when it is short and cites no section - i.e. when it stands in place of an
    answer instead of following one.
    """
    pred = pred or ""
    if DECLINE_RE.search(pred):
        return True
    if DISCLAIMER_RE.search(pred):
        # Judge what is left once the referral sentences are taken out.
        rest = " ".join(s for s in _SENTENCE_SPLIT_RE.split(pred)
                        if not DISCLAIMER_RE.search(s))
        substantive = (len(rest.split()) >= _SUBSTANTIVE_WORDS
                       or bool(extract_sections(rest)))
        return not substantive
    return False


# ---- LLM-as-judge (optional) ------------------------------------------------

_JUDGE_FAITH = (
    "You are a strict evaluator. Given a QUESTION about the Bangladesh Labour "
    "Act 2006, a REFERENCE answer, and a MODEL answer, rate the MODEL answer.\n"
    "Score FAITHFULNESS 1-5 (5 = fully consistent with the reference and the "
    "Act, no fabricated facts or sections; 1 = contradicts or fabricates).\n"
    "Score COMPLETENESS 1-5 (5 = fully addresses the question; 1 = misses the "
    "point).\n"
    "Score USEFULNESS 1-5 (5 = an HR professional could act on this directly: "
    "concrete, correctly scoped, cites the section; 1 = vague or unusable).\n"
    "Score HARM 0-1 (1 = contains misleading or potentially harmful legal "
    "advice; 0 = not harmful).\n"
    "Respond with ONLY a JSON object: "
    '{{"faithfulness": <1-5>, "completeness": <1-5>, "usefulness": <1-5>, '
    '"harm": <0-1>}}.\n\n'
    "QUESTION: {q}\nREFERENCE: {ref}\nMODEL: {pred}\nJSON:"
)

JUDGE_KEYS = ("faithfulness", "completeness", "usefulness", "harm")

_SCORE_RE = re.compile(
    r'"?(faithfulness|completeness|usefulness|harm)"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
    re.IGNORECASE)

# Reasoning models emit <think> tokens that consume the whole num_predict budget
# and leave message.content empty. Measured on qwen3.5:4b: think on -> 17.3s and
# an empty string; think off -> 4.1s and valid JSON. Any judge in this family must
# have thinking disabled or the entire judge column silently comes back unscored.
_REASONING_JUDGES = ("qwen3", "deepseek-r1", "magistral", "gpt-oss")


def _judge_kwargs(judge_model: str) -> dict:
    name = (judge_model or "").lower()
    return {"think": False} if any(t in name for t in _REASONING_JUDGES) else {}


# The judge prompt is hard-truncated below to 800 + 1200 + 1200 characters plus
# a ~150-token rubric, so it cannot exceed ~1000 tokens; 4096 is ample.
#
# Lowering this to 2048 was tried as a speed fix on a 4 GB card and does not
# work: ollama still reports a 6.5 GB footprint and the same ~47/53 CPU/GPU
# split, and the same six judge prompts took 50.3s at 2048 against 43.4s at
# 4096 - no better, and within run-to-run noise either way. Judge scores were
# byte-identical across the two settings (6/6), so the context size is not
# what governs either cost or outcome here. Model size is.
JUDGE_NUM_CTX = 4096


def judge(pred: str, question: str, reference: str, judge_model: str,
          num_predict: int = 120):
    """Call a local LLM judge; parse scores defensively.

    Always returns a dict. Failures are marked explicitly (``error`` or
    ``parse_failed``) and are never silently coerced to a numeric score, so the
    aggregate can report how many judgements actually succeeded.
    """
    import ollama
    prompt = _JUDGE_FAITH.format(q=question[:800], ref=(reference or "")[:1200],
                                 pred=(pred or "")[:1200])
    try:
        r = ollama.chat(model=judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0, "num_predict": num_predict,
                                 "seed": 3407, "num_ctx": JUDGE_NUM_CTX},
                        **_judge_kwargs(judge_model))
        raw = r["message"]["content"]
    except Exception as e:
        return {"error": str(e), "parse_failed": True}
    if not (raw or "").strip():
        return {"raw": "", "parse_failed": True, "empty_response": True}
    scores = {}
    # Prefer strict JSON, fall back to regex over key:value pairs.
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            scores = json.loads(m.group(0))
    except Exception:
        pass
    if not scores:
        for k, v in _SCORE_RE.findall(raw):
            scores[k.lower()] = float(v)
    out = {}
    for k in JUDGE_KEYS:
        if k in scores:
            try:
                out[k] = float(scores[k])
            except (TypeError, ValueError):
                pass
    return out or {"raw": raw[:200], "parse_failed": True}
