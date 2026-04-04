import pytest

from bitcal_tts.calibrator.bit_aware import (
    BitAwareCalibrator,
    calibrate_confidence,
    effective_scale_for_bit_width,
)


def test_calibrate_in_unit_interval():
    c = calibrate_confidence(
        entropy=2.0,
        trace_stability=0.9,
        hidden_stability=0.9,
        bit_width=8,
        max_entropy=10.0,
    )
    assert 0.0 <= c <= 1.0


def test_lower_bit_more_conservative():
    c4 = calibrate_confidence(
        entropy=5.0,
        trace_stability=0.5,
        hidden_stability=0.5,
        bit_width=4,
    )
    c16 = calibrate_confidence(
        entropy=5.0,
        trace_stability=0.5,
        hidden_stability=0.5,
        bit_width=16,
    )
    assert c4 <= c16 + 1e-6


def test_bit_aware_calibrator_callable():
    cal = BitAwareCalibrator(bit_width=8)
    v = cal(entropy=3.0, trace_stability=0.8, hidden_stability=0.8)
    assert 0.0 <= v <= 1.0


def test_effective_scale_branches():
    assert effective_scale_for_bit_width(4) == 0.85
    assert effective_scale_for_bit_width(8) == 1.0
    assert effective_scale_for_bit_width(16) == 1.05


def test_bit_aware_effective_scale_instance_method():
    assert BitAwareCalibrator(bit_width=4).effective_scale() == pytest.approx(0.85)


def test_temperature_sharpens_confidence():
    base = calibrate_confidence(
        entropy=2.0,
        trace_stability=0.5,
        hidden_stability=0.5,
        bit_width=8,
        temperature=1.0,
        max_entropy=10.0,
    )
    warm = calibrate_confidence(
        entropy=2.0,
        trace_stability=0.5,
        hidden_stability=0.5,
        bit_width=8,
        temperature=2.0,
        max_entropy=10.0,
    )
    assert base != warm
    assert 0.0 <= warm <= 1.0


def test_explicit_scale_overrides_bit_width():
    c = calibrate_confidence(
        entropy=0.0,
        trace_stability=1.0,
        hidden_stability=1.0,
        bit_width=4,
        scale=1.0,
        max_entropy=10.0,
    )
    assert c == pytest.approx(1.0, abs=1e-5)


def test_weights_renormalized():
    c = calibrate_confidence(
        entropy=0.0,
        trace_stability=1.0,
        hidden_stability=0.0,
        w_entropy=2.0,
        w_trace=2.0,
        w_hidden=0.0,
        max_entropy=10.0,
    )
    assert c == pytest.approx(1.0, abs=1e-5)


def test_invalid_weights_raises():
    with pytest.raises(ValueError, match="Weights"):
        calibrate_confidence(0.0, 1.0, 1.0, w_entropy=-1.0, w_trace=1.0, w_hidden=1.0)


def test_invalid_max_entropy_raises():
    with pytest.raises(ValueError, match="max_entropy"):
        calibrate_confidence(0.0, 1.0, 1.0, max_entropy=0.0)


def test_invalid_temperature_raises():
    with pytest.raises(ValueError, match="temperature"):
        calibrate_confidence(0.0, 1.0, 1.0, temperature=0.0)


def test_invalid_bit_width_raises():
    with pytest.raises(ValueError, match="bit_width"):
        effective_scale_for_bit_width(0)


def test_bit_aware_bad_temperature_in_dataclass():
    with pytest.raises(ValueError, match="temperature"):
        BitAwareCalibrator(temperature=0.0)


def test_bit_aware_bad_weights_in_dataclass():
    with pytest.raises(ValueError, match="Weights"):
        BitAwareCalibrator(w_entropy=1.0, w_trace=-0.5, w_hidden=0.5)
