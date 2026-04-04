"""
GSM8K benchmark loader for BitCal-TTS experiments.

Supports two sources:
  1. Hugging Face ``datasets`` library  (pip install datasets) — primary.
  2. Local JSONL fallback with fields: id, prompt, answer.

Answer extraction handles the canonical ``#### <number>`` format and a
numeric-last-line fallback so the same function works for other math datasets.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

# Canonical prompt template used for all methods (change here = change everywhere)
GSM8K_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Show your reasoning clearly.\n"
    "At the end of your solution, write the final numeric answer as: #### <answer>\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)


@dataclass
class GSM8KItem:
    id: str
    question: str
    gold_answer: Optional[str]   # normalized numeric string, e.g. "80"
    prompt: str                  # fully formatted prompt for the model


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANS_PATTERN = re.compile(r"####\s*([0-9,\-\.]+)")
_NUM_PATTERN = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from model output.

    Tries:
    1. ``#### <number>`` (canonical GSM8K format).
    2. Last number in the text (fallback).
    """
    m = _ANS_PATTERN.search(text)
    if m:
        return _normalize_number(m.group(1))
    nums = _NUM_PATTERN.findall(text)
    return _normalize_number(nums[-1]) if nums else None


def _normalize_number(s: str) -> str:
    """Strip commas and trailing zeros; convert to plain numeric string."""
    s = s.replace(",", "").strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    """Numeric equality check; falls back to string equality."""
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()


# ---------------------------------------------------------------------------
# HF datasets loader
# ---------------------------------------------------------------------------

def _datasets_available() -> bool:
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


def load_gsm8k_hf(
    split: str = "test",
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> List[GSM8KItem]:
    """
    Load GSM8K from Hugging Face ``datasets`` library.

    Parameters
    ----------
    split      : "train" or "test".
    n_samples  : if set, take a deterministic subset (first N after fixed shuffle).
    seed       : shuffle seed for reproducibility.
    """
    if not _datasets_available():
        raise ImportError("Install datasets: pip install datasets")

    from datasets import load_dataset  # type: ignore

    ds = load_dataset("openai/gsm8k", "main", split=split)

    if n_samples is not None and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))

    items: List[GSM8KItem] = []
    for i, row in enumerate(ds):
        question = row["question"]
        # GSM8K gold answer contains reasoning + "#### <number>"
        gold_text = row.get("answer", "")
        gold = extract_answer(gold_text)
        items.append(
            GSM8KItem(
                id=f"gsm8k-{split}-{i:05d}",
                question=question,
                gold_answer=gold,
                prompt=GSM8K_PROMPT_TEMPLATE.format(problem=question),
            )
        )
    return items


# ---------------------------------------------------------------------------
# JSONL fallback loader
# ---------------------------------------------------------------------------

def load_gsm8k_jsonl(
    path: str | Path,
    n_samples: Optional[int] = None,
) -> List[GSM8KItem]:
    """
    Load from a JSONL file.  Expected fields: ``prompt`` (required),
    optionally ``id`` and ``answer`` (gold numeric string).
    """
    p = Path(path)
    items: List[GSM8KItem] = []
    with p.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question", row.get("prompt", ""))
            gold_text = row.get("answer", "")
            gold = extract_answer(str(gold_text)) if gold_text else gold_text or None
            items.append(
                GSM8KItem(
                    id=str(row.get("id", idx)),
                    question=question,
                    gold_answer=gold,
                    prompt=row.get("full_prompt") or GSM8K_PROMPT_TEMPLATE.format(problem=question),
                )
            )
            if n_samples is not None and len(items) >= n_samples:
                break
    return items


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_gsm8k(
    source: str = "hf",
    *,
    split: str = "test",
    n_samples: Optional[int] = None,
    seed: int = 42,
    jsonl_path: Optional[str | Path] = None,
) -> List[GSM8KItem]:
    """
    Unified entry point.

    Parameters
    ----------
    source      : "hf" (Hugging Face datasets) or "jsonl".
    split       : HF split if source=="hf".
    n_samples   : cap number of items.
    seed        : reproducibility seed (for HF shuffle).
    jsonl_path  : required when source=="jsonl".
    """
    if source == "hf":
        return load_gsm8k_hf(split=split, n_samples=n_samples, seed=seed)
    if source == "jsonl":
        if jsonl_path is None:
            raise ValueError("jsonl_path required when source='jsonl'")
        return load_gsm8k_jsonl(jsonl_path, n_samples=n_samples)
    raise ValueError(f"Unknown source {source!r}; use 'hf' or 'jsonl'")


def iter_gsm8k(
    source: str = "hf",
    **kwargs: object,
) -> Iterator[GSM8KItem]:
    yield from load_gsm8k(source=source, **kwargs)
