"""Hidden-state drift between reasoning steps (optional signal)."""

from __future__ import annotations

from typing import List, Optional

import torch


def hidden_state_stability(
    hiddens: List[Optional[torch.Tensor]],
) -> float:
    """
    Mean cosine similarity between consecutive non-None hidden states.

    Returns 1.0 if fewer than two usable tensors.
    """
    vecs: List[torch.Tensor] = []
    for h in hiddens:
        if h is None:
            continue
        v = h.detach().float().reshape(-1)
        if v.numel() == 0:
            continue
        vecs.append(v / (v.norm() + 1e-8))
    if len(vecs) < 2:
        return 1.0
    sims = []
    for a, b in zip(vecs[:-1], vecs[1:]):
        m = min(a.numel(), b.numel())
        if m == 0:
            continue
        sims.append(float((a[:m] * b[:m]).sum().item()))
    if not sims:
        return 1.0
    return sum(sims) / len(sims)
