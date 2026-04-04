"""Evaluation metrics for BitCal-TTS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from bitcal_tts.runner.baseline import ReasoningTrace
from bitcal_tts.signals.entropy import token_entropy
from bitcal_tts.signals.hidden import hidden_state_stability
from bitcal_tts.signals.trace import reasoning_trace_stability


@dataclass
class TraceSummary:
    n_steps: int
    total_tokens: int
    mean_entropy: float
    trace_stability: float
    hidden_stability: float


def summarize_trace(trace: ReasoningTrace) -> TraceSummary:
    entropies = [token_entropy(l) for l in trace.logits_per_step]
    mean_e = sum(entropies) / len(entropies) if entropies else 0.0
    return TraceSummary(
        n_steps=len(trace.texts),
        total_tokens=trace.total_tokens,
        mean_entropy=mean_e,
        trace_stability=reasoning_trace_stability(trace.texts),
        hidden_stability=hidden_state_stability(trace.hidden_states),
    )


def halting_metrics(
    actions: List[str],
    final_answer_correct: bool,
    tokens_used: int,
    budget: int,
) -> Dict[str, Any]:
    """Paper-oriented metrics dict."""
    return {
        "accuracy": 1.0 if final_answer_correct else 0.0,
        "tokens_used": tokens_used,
        "budget": budget,
        "token_efficiency": tokens_used / max(budget, 1),
        "n_escalations": sum(1 for a in actions if a == "escalate"),
        "n_stops": sum(1 for a in actions if a == "stop"),
    }
