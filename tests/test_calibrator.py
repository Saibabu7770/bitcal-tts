from bitcal_tts.calibrator.bit_aware import BitAwareCalibrator, calibrate_confidence


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
