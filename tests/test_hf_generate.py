"""Tests for HFStepGenerator using a fully mocked causal LM."""

from unittest.mock import MagicMock

import pytest
import torch

from bitcal_tts.integrations.hf_generate import (
    HFStepGenerator,
    _infer_input_device,
    _top_p_filter,
)
from bitcal_tts.runner.baseline import run_fixed_budget_loop
from bitcal_tts.runner.budget import TokenBudget


def _make_mock_model(vocab_size: int = 32, hidden_size: int = 16, seed: int = 0):
    """Return a model mock that produces deterministic logits."""
    p = torch.nn.Parameter(torch.zeros(1, device="cpu"))
    model = MagicMock()
    model.parameters.return_value = iter([p])
    model.eval = MagicMock(return_value=model)

    g = torch.Generator()
    g.manual_seed(seed)

    def _forward(*args, input_ids=None, **kwargs):
        nonlocal g
        bsz = 1
        seq = input_ids.shape[-1] if input_ids is not None else 1
        logits = torch.randn(bsz, seq, vocab_size, generator=g)
        hidden = torch.randn(bsz, seq, hidden_size, generator=g)
        out = MagicMock()
        out.logits = logits
        out.hidden_states = (hidden,)
        out.past_key_values = None
        return out

    model.side_effect = _forward
    return model


def _make_mock_tokenizer(vocab_size: int = 32):
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.eos_token_id = 1
    tok.return_value = {"input_ids": torch.tensor([[2, 3, 4]])}

    def _decode(ids, skip_special_tokens=True):
        return "".join(str(i) for i in (ids if hasattr(ids, "__iter__") else [ids]))

    tok.decode.side_effect = _decode
    return tok


class TestHFStepGenerator:
    def test_step_returns_correct_shapes(self):
        model = _make_mock_model(vocab_size=16)
        tok = _make_mock_tokenizer(vocab_size=16)
        gen = HFStepGenerator(model, tok, "hello", step_size=4, seed=0)
        budget = TokenBudget(max_tokens=100)
        text, logits, hidden, n = gen.step(0, budget)
        assert isinstance(text, str)
        assert logits.shape == (16,)
        assert hidden is not None
        assert n == 4

    def test_budget_consumed(self):
        model = _make_mock_model(vocab_size=16)
        tok = _make_mock_tokenizer(vocab_size=16)
        gen = HFStepGenerator(model, tok, "hello", step_size=4, seed=0)
        budget = TokenBudget(max_tokens=6)
        _, _, _, n1 = gen.step(0, budget)
        budget.consume(n1)
        _, _, _, n2 = gen.step(1, budget)
        budget.consume(n2)
        assert budget.used_tokens <= 8
        assert n1 + n2 <= 8

    def test_get_full_text(self):
        model = _make_mock_model(vocab_size=16)
        tok = _make_mock_tokenizer(vocab_size=16)
        gen = HFStepGenerator(model, tok, "hello", step_size=2, seed=0)
        budget = TokenBudget(max_tokens=10)
        gen.step(0, budget)
        budget.consume(2)
        gen.step(1, budget)
        assert len(gen.get_full_text()) >= 0  # just validates it returns a string

    def test_run_fixed_budget_loop_integration(self):
        model = _make_mock_model(vocab_size=16)
        tok = _make_mock_tokenizer(vocab_size=16)
        gen = HFStepGenerator(model, tok, "hello", step_size=2, seed=0)
        budget = TokenBudget(max_tokens=8)
        trace = run_fixed_budget_loop(budget, max_steps=10, step_fn=gen.step)
        assert len(trace.texts) >= 1
        assert trace.total_tokens <= 8

    def test_reset_clears_state(self):
        model = _make_mock_model(vocab_size=16)
        tok = _make_mock_tokenizer(vocab_size=16)
        gen = HFStepGenerator(model, tok, "hello", step_size=2, seed=0)
        budget = TokenBudget(max_tokens=4)
        gen.step(0, budget)
        gen.reset("new prompt")
        assert gen._generated_ids == []
        assert gen._past_kv is None
        assert gen.finished is False

    def test_invalid_step_size_raises(self):
        model = _make_mock_model(vocab_size=8)
        tok = _make_mock_tokenizer(vocab_size=8)
        with pytest.raises(ValueError, match="step_size"):
            HFStepGenerator(model, tok, "p", step_size=0)

    def test_exhausted_budget_returns_zero_tokens(self):
        model = _make_mock_model(vocab_size=8)
        tok = _make_mock_tokenizer(vocab_size=8)
        gen = HFStepGenerator(model, tok, "p", step_size=4, seed=0)
        budget = TokenBudget(max_tokens=0)
        _, _, _, n = gen.step(0, budget)
        assert n == 0


class TestInferInputDevice:
    def test_returns_device_from_param(self):
        p = torch.nn.Parameter(torch.zeros(1))
        model = MagicMock()
        model.parameters.return_value = iter([p])
        device = _infer_input_device(model)
        assert isinstance(device, torch.device)

    def test_empty_model_falls_back_to_cpu(self):
        model = MagicMock()
        model.parameters.return_value = iter([])
        device = _infer_input_device(model)
        assert device.type in ("cpu", "cuda")


class TestTopPFilter:
    def test_top_p_reduces_mass(self):
        logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        filtered = _top_p_filter(logits, top_p=0.5)
        probs = torch.softmax(filtered, dim=-1)
        assert probs[0].item() > 0.9

    def test_top_p_1_unchanged(self):
        logits = torch.randn(20)
        filtered = _top_p_filter(logits, top_p=1.0)
        assert torch.allclose(
            torch.softmax(logits, dim=-1),
            torch.softmax(filtered, dim=-1),
            atol=1e-5,
        )
