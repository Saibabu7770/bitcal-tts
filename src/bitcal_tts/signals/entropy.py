"""Entropy from next-token logits (calibration / halting signal)."""

from __future__ import annotations

import torch


def token_entropy(logits: torch.Tensor, dim: int = -1) -> float:
    """
    Shannon entropy of softmax(logits) in nats.

    logits: shape [..., V] or [V]; returns scalar float.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    log_probs = torch.log_softmax(logits.float(), dim=dim)
    probs = log_probs.exp()
    ent = -(probs * log_probs).sum(dim=dim)
    return float(ent.mean().item())
