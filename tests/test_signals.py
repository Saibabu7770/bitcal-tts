import torch

from bitcal_tts.signals.entropy import token_entropy
from bitcal_tts.signals.hidden import hidden_state_stability
from bitcal_tts.signals.trace import reasoning_trace_stability


def test_token_entropy_uniform_high():
    logits = torch.zeros(100)
    e = token_entropy(logits)
    assert e > 4.0


def test_token_entropy_peaked_low():
    logits = torch.full((100,), -10.0)
    logits[0] = 10.0
    e = token_entropy(logits)
    assert e < 1.0


def test_trace_stability_identical():
    texts = ["hello world " * 2, "hello world " * 2]
    assert reasoning_trace_stability(texts, min_chars=4) == 1.0


def test_hidden_stability():
    a = torch.randn(64)
    b = a.clone()
    s = hidden_state_stability([a, b])
    assert s > 0.99


def test_hidden_skips_none_and_empty():
    assert hidden_state_stability([None, None]) == 1.0
    assert hidden_state_stability([torch.tensor([]), torch.tensor([])]) == 1.0


def test_hidden_mismatched_lengths():
    a = torch.randn(4)
    b = torch.randn(8)
    s = hidden_state_stability([a, b])
    assert -1.0 <= s <= 1.0
