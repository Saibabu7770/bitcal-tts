import pytest

from bitcal_tts.policy.halting import HaltingAction, HaltingPolicy


def test_stop_low_entropy_high_conf():
    p = HaltingPolicy(
        stop_entropy_threshold=3.0,
        escalate_entropy_threshold=6.0,
        min_budget_to_continue=8,
        stop_confidence_threshold=0.5,
    )
    assert p.decide(entropy=1.0, confidence=0.9, budget_remaining=100) == HaltingAction.STOP


def test_escalate_high_entropy():
    p = HaltingPolicy(escalate_entropy_threshold=4.0, min_budget_to_continue=8)
    assert p.decide(entropy=5.0, confidence=0.5, budget_remaining=100) == HaltingAction.ESCALATE


def test_stop_low_budget():
    p = HaltingPolicy(min_budget_to_continue=32)
    assert p.decide(entropy=3.0, confidence=0.9, budget_remaining=10) == HaltingAction.STOP


def test_continue_between_thresholds():
    """Entropy above stop band but below escalate; confidence too low for early stop."""
    p = HaltingPolicy(
        stop_entropy_threshold=1.0,
        escalate_entropy_threshold=10.0,
        min_budget_to_continue=8,
        stop_confidence_threshold=0.99,
    )
    assert p.decide(entropy=3.0, confidence=0.5, budget_remaining=100) == HaltingAction.CONTINUE


def test_invalid_escalate_below_stop_raises():
    with pytest.raises(ValueError, match="escalate_entropy_threshold"):
        HaltingPolicy(stop_entropy_threshold=5.0, escalate_entropy_threshold=4.0)


def test_invalid_min_budget_raises():
    with pytest.raises(ValueError, match="min_budget_to_continue"):
        HaltingPolicy(min_budget_to_continue=-1)
