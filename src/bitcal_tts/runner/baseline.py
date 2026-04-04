"""
Fixed-budget reasoning loop skeleton.

For research integration, plug in HF generate() or vLLM; tests use mock logits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from bitcal_tts.runner.budget import TokenBudget


@dataclass
class ReasoningTrace:
    """One reasoning run: text chunks, logits, optional hidden states."""

    texts: List[str] = field(default_factory=list)
    logits_per_step: List[torch.Tensor] = field(default_factory=list)
    hidden_states: List[Optional[torch.Tensor]] = field(default_factory=list)
    total_tokens: int = 0

    def append_step(
        self,
        text: str,
        logits: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        tokens_this_step: int = 1,
    ) -> None:
        self.texts.append(text)
        self.logits_per_step.append(logits)
        self.hidden_states.append(hidden)
        self.total_tokens += tokens_this_step


def run_fixed_budget_loop(
    budget: TokenBudget,
    max_steps: int,
    step_fn: Callable[[int, TokenBudget], tuple[str, torch.Tensor, Optional[torch.Tensor], int]],
) -> ReasoningTrace:
    """
    Run up to max_steps; each step_fn returns (text, logits, hidden_state, n_tokens).

    Stops when budget exhausted or max_steps reached.
    """
    trace = ReasoningTrace()
    for step in range(max_steps):
        if budget.exhausted():
            break
        text, logits, hidden, n_tok = step_fn(step, budget)
        trace.append_step(text, logits, hidden, n_tok)
        budget.consume(n_tok)
    return trace


def mock_step_fn(
    vocab_size: int = 32000,
    device: Optional[torch.device] = None,
) -> Callable[[int, TokenBudget], tuple[str, torch.Tensor, Optional[torch.Tensor], int]]:
    """Deterministic mock for tests: returns logits + fake hidden state."""

    if device is None:
        device = torch.device("cpu")
    g = torch.Generator(device=device)
    g.manual_seed(0)

    def _fn(step: int, _b: TokenBudget) -> tuple[str, torch.Tensor, Optional[torch.Tensor], int]:
        g.manual_seed(1000 + step)
        logits = torch.randn(vocab_size, device=device, generator=g) * 0.1
        hidden = torch.randn(1, 128, device=device, generator=g)
        return f"chunk_{step}", logits, hidden, 8

    return _fn
