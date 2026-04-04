"""Text-level reasoning trace stability (cheap proxy for answer convergence)."""

from __future__ import annotations

from typing import List


def reasoning_trace_stability(texts: List[str], min_chars: int = 8) -> float:
    """
    Rough stability in [0, 1]: 1 if consecutive chunks are identical after strip.

    If fewer than 2 chunks, returns 1.0.
    """
    if len(texts) < 2:
        return 1.0
    stable = 0
    pairs = 0
    for a, b in zip(texts[:-1], texts[1:]):
        sa, sb = a.strip(), b.strip()
        if len(sa) < min_chars or len(sb) < min_chars:
            continue
        pairs += 1
        if sa == sb:
            stable += 1
    if pairs == 0:
        return 1.0
    return stable / pairs
