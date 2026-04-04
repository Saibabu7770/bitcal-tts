"""Adaptive halting / escalation policy under token budget."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HaltingAction(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"
    ESCALATE = "escalate"  # e.g. more compute / full precision (hook for deployment)


@dataclass
class HaltingPolicy:
    stop_entropy_threshold: float = 2.0
    escalate_entropy_threshold: float = 4.0
    min_budget_to_continue: int = 32
    stop_confidence_threshold: float = 0.75

    def decide(
        self,
        entropy: float,
        confidence: float,
        budget_remaining: int,
    ) -> HaltingAction:
        if budget_remaining < self.min_budget_to_continue:
            return HaltingAction.STOP
        if entropy >= self.escalate_entropy_threshold:
            return HaltingAction.ESCALATE
        if entropy <= self.stop_entropy_threshold and confidence >= self.stop_confidence_threshold:
            return HaltingAction.STOP
        return HaltingAction.CONTINUE
