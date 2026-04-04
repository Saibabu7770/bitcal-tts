"""Token / step budget tracking for reasoning loops."""

from dataclasses import dataclass


@dataclass
class TokenBudget:
    """Remaining budget for adaptive reasoning."""

    max_tokens: int
    used_tokens: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def consume(self, n: int) -> None:
        self.used_tokens += max(0, n)

    def exhausted(self) -> bool:
        return self.remaining <= 0
