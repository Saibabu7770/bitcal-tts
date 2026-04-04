"""
Bit-aware confidence calibration (research hook).

Maps raw signals (entropy, trace stability, hidden stability) to a scalar
confidence using bit-width as a scaling factor (lower precision -> more conservative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


def effective_scale_for_bit_width(bit_width: int) -> float:
    """
    Map nominal weight bit width to a multiplicative confidence scale.

    Lower effective precision uses a slightly conservative scale (< 1).
    """
    if bit_width < 1:
        raise ValueError(f"bit_width must be >= 1, got {bit_width}")
    if bit_width <= 4:
        return 0.85
    if bit_width <= 8:
        return 1.0
    return 1.05


def _normalize_weights(
    w_entropy: float,
    w_trace: float,
    w_hidden: float,
) -> Tuple[float, float, float]:
    s = w_entropy + w_trace + w_hidden
    if not (s > 0.0) or not all(x >= 0.0 for x in (w_entropy, w_trace, w_hidden)):
        raise ValueError(
            "Weights must be non-negative and sum to a positive value; "
            f"got w_entropy={w_entropy}, w_trace={w_trace}, w_hidden={w_hidden}"
        )
    return w_entropy / s, w_trace / s, w_hidden / s


@dataclass
class BitAwareCalibrator:
    """Linear blend of signals scaled by effective precision."""

    bit_width: int = 8
    temperature: float = 1.0
    w_entropy: float = 0.4
    w_trace: float = 0.35
    w_hidden: float = 0.25

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        _normalize_weights(self.w_entropy, self.w_trace, self.w_hidden)

    def effective_scale(self) -> float:
        """Higher bit width -> closer to 1.0; 4-bit -> more conservative."""
        return effective_scale_for_bit_width(self.bit_width)

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

    If scale is None, uses effective_scale_for_bit_width(bit_width).
    Weights are renormalized to sum to 1 for numerical robustness.
    """
    we, wt, wh = _normalize_weights(w_entropy, w_trace, w_hidden)
    if scale is None:
        scale = effective_scale_for_bit_width(bit_width)
    if max_entropy <= 0:
        raise ValueError(f"max_entropy must be > 0, got {max_entropy}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    norm_ent = min(1.0, max(0.0, entropy / max_entropy))
    conf_raw = we * (1.0 - norm_ent) + wt * trace_stability + wh * hidden_stability
    conf = max(0.0, min(1.0, conf_raw * scale))
    if temperature != 1.0:
        conf = max(0.0, min(1.0, conf ** (1.0 / temperature)))
    return conf
