"""Hidden-state drift between reasoning steps (optional signal)."""

from __future__ import annotations

from typing import List, Optional

import torch


def hidden_state_stability(
    hiddens: List[Optional[torch.Tensor]],
) -> float:
    """
    Mean cosine similarity between consecutive non-None hidden states.

    Returns 1.0 if fewer than two usable tensors. Empty tensors are skipped.
    """
    vecs: List[torch.Tensor] = []
    for h in hiddens:
        if h is None:
            continue
        v = h.detach().float().reshape(-1)
        if v.numel() == 0:
            continue
        n = v.norm()
        vecs.append(v / (n + 1e-8))
    if len(vecs) < 2:
        return 1.0
    sims: List[float] = []
    for a, b in zip(vecs[:-1], vecs[1:]):
        m = min(a.numel(), b.numel())
        sims.append(float((a[:m] * b[:m]).sum().item()))
    return sum(sims) / len(sims)
