"""Tests for src/bitcal_tts/experiment.py (run_item, run_fixed_method, run_adaptive_method)."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from bitcal_tts.experiment import METHODS, ItemResult, run_item
from bitcal_tts.policy.halting import HaltingPolicy


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

VOCAB = 16
HIDDEN = 8


def _make_model(vocab_size=VOCAB, hidden_size=HIDDEN, seed=0):
    p = torch.nn.Parameter(torch.zeros(1))
    model = MagicMock()
    model.parameters.return_value = iter([p])
    model.eval = MagicMock(return_value=model)
    g = torch.Generator()
    g.manual_seed(seed)

    def _forward(*args, input_ids=None, **kwargs):
        nonlocal g
        seq = input_ids.shape[-1] if input_ids is not None else 1
        logits = torch.randn(1, seq, vocab_size, generator=g)
        hidden = torch.randn(1, seq, hidden_size, generator=g)
        out = MagicMock()
        out.logits = logits
        out.hidden_states = (hidden,)
        out.past_key_values = None
        return out

    model.side_effect = _forward
    return model


def _make_tokenizer(vocab_size=VOCAB, answer_text="So the answer is #### 42"):
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.eos_token_id = 1

    def _call(text, return_tensors=None, **kwargs):
        ids = torch.tensor([[2, 3, 4]])
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}

    tok.side_effect = _call

    def _decode(ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if len(ids) > 0:
            return answer_text
        return ""

    tok.decode.side_effect = _decode
    return tok


def _make_policy():
    return HaltingPolicy(
        stop_entropy_threshold=0.0,
        escalate_entropy_threshold=100.0,
        stop_confidence_threshold=0.0,
        min_budget_to_continue=0,
    )


def _run_item_common(method, budget=32):
    model = _make_model()
    tok = _make_tokenizer()

    # For fixed method: model.generate must return proper tensor
    input_len = 3
    gen_len = min(budget, 8)
    gen_ids = torch.randint(2, VOCAB, (1, input_len + gen_len))
    model.generate = MagicMock(return_value=gen_ids)

    return run_item(
        model=model,
        tokenizer=tok,
        task_id="test-001",
        prompt="What is 6*7?",
        gold_answer="42",
        method=method,
        budget=budget,
        bit_width=4,
        policy=_make_policy(),
        seed=42,
        step_size=4,
        max_entropy=10.0,
        answer_extractor=lambda t: "42",
    )


class TestRunItemFixed:
    def test_returns_item_result(self):
        result = _run_item_common("fixed")
        assert isinstance(result, ItemResult)

    def test_method_is_fixed(self):
        result = _run_item_common("fixed")
        assert result.method == "fixed"

    def test_correct_answer(self):
        result = _run_item_common("fixed")
        assert result.correct is True

    def test_budget_stored(self):
        result = _run_item_common("fixed", budget=64)
        assert result.budget == 64

    def test_to_dict(self):
        result = _run_item_common("fixed")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["method"] == "fixed"
        assert "correct" in d

    def test_incorrect_gold_answer(self):
        model = _make_model()
        tok = _make_tokenizer()
        gen_ids = torch.randint(2, VOCAB, (1, 3 + 8))
        model.generate = MagicMock(return_value=gen_ids)
        result = run_item(
            model=model, tokenizer=tok,
            task_id="x", prompt="q", gold_answer="999",
            method="fixed", budget=16, bit_width=4,
            policy=_make_policy(), seed=0, step_size=4,
            max_entropy=10.0, answer_extractor=lambda t: "42",
        )
        assert result.correct is False


class TestRunItemAdaptive:
    def test_adaptive_returns_item_result(self):
        result = _run_item_common("adaptive")
        assert isinstance(result, ItemResult)
        assert result.method == "adaptive"

    def test_bitcal_tts_returns_item_result(self):
        result = _run_item_common("bitcal_tts")
        assert isinstance(result, ItemResult)
        assert result.method == "bitcal_tts"

    def test_adaptive_correct_answer(self):
        result = _run_item_common("adaptive")
        assert result.correct is True

    def test_tokens_used_within_budget(self):
        result = _run_item_common("adaptive", budget=16)
        assert result.tokens_used <= 16

    def test_actions_list_present(self):
        result = _run_item_common("adaptive")
        assert isinstance(result.actions, list)

    def test_bit_width_stored(self):
        result = _run_item_common("bitcal_tts")
        assert result.bit_width == 4

    def test_invalid_method_raises(self):
        model = _make_model()
        tok = _make_tokenizer()
        model.generate = MagicMock(return_value=torch.zeros(1, 5, dtype=torch.long))
        with pytest.raises(ValueError, match="Unsupported method"):
            run_item(
                model=model, tokenizer=tok,
                task_id="x", prompt="q", gold_answer="1",
                method="unknown_method", budget=16, bit_width=4,
                policy=_make_policy(), seed=0, step_size=4,
                max_entropy=10.0, answer_extractor=lambda t: "1",
            )


class TestMethodsConstant:
    def test_methods_tuple(self):
        assert "fixed" in METHODS
        assert "adaptive" in METHODS
        assert "bitcal_tts" in METHODS
        assert len(METHODS) == 3
