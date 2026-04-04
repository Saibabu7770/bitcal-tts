"""
Bit-aware confidence calibration (research hook).

Maps raw signals (entropy, trace stability, hidden stability) to a scalar
confidence using bit-width as a scaling factor (lower precision -> more conservative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BitAwareCalibrator:
    """Linear blend of signals scaled by effective precision."""

    bit_width: int = 8
    temperature: float = 1.0
    w_entropy: float = 0.4
    w_trace: float = 0.35
    w_hidden: float = 0.25

    def effective_scale(self) -> float:
        """Higher bit width -> closer to 1.0; 4-bit -> more conservative."""
        # Map 4,8,16 -> ~0.85, 1.0, 1.05
        if self.bit_width <= 4:
            return 0.85
        if self.bit_width <= 8:
            return 1.0
        return 1.05

    def __call__(
        self,
        entropy: float,
        trace_stability: float,
        hidden_stability: float,
        max_entropy: float = 10.0,
    ) -> float:
        return calibrate_confidence(
            entropy=entropy,
            trace_stability=trace_stability,
            hidden_stability=hidden_stability,
            bit_width=self.bit_width,
            temperature=self.temperature,
            w_entropy=self.w_entropy,
            w_trace=self.w_trace,
            w_hidden=self.w_hidden,
            max_entropy=max_entropy,
        )


def calibrate_confidence(
    entropy: float,
    trace_stability: float,
    hidden_stability: float,
    bit_width: int = 8,
    temperature: float = 1.0,
    w_entropy: float = 0.4,
    w_trace: float = 0.35,
    w_hidden: float = 0.25,
    max_entropy: float = 10.0,
    scale: Optional[float] = None,
) -> float:
    """
    Return confidence in [0, 1]: higher when entropy is low and stabilities high.

    If scale is None, uses BitAwareCalibrator(bit_width).effective_scale().
    """
    if scale is None:
        scale = BitAwareCalibrator(bit_width=bit_width).effective_scale()
    norm_ent = min(1.0, max(0.0, entropy / max(max_entropy, 1e-6)))
    conf_raw = (
        w_entropy * (1.0 - norm_ent)
        + w_trace * trace_stability
        + w_hidden * hidden_stability
    )
    conf = max(0.0, min(1.0, conf_raw * scale))
    if temperature != 1.0 and temperature > 0:
        conf = max(0.0, min(1.0, conf ** (1.0 / temperature)))
    return conf
