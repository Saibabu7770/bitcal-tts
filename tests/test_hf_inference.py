from unittest.mock import MagicMock

import pytest
import torch

from bitcal_tts.integrations import hf_inference as hf
from bitcal_tts.integrations.hf_inference import (
    forward_last_logits,
    hf_smoke_forward,
    hf_smoke_report,
    load_causal_lm,
    transformers_available,
)


def test_transformers_flag_is_bool():
    assert isinstance(transformers_available(), bool)


def test_forward_last_logits_without_hidden_states():
    p = torch.nn.Parameter(torch.zeros(1, device="cpu"))
    model = MagicMock()
    model.parameters.return_value = iter([p])
    logits = torch.randn(1, 4, 10)
    out = MagicMock()
    out.logits = logits
    out.hidden_states = None
    model.return_value = out
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
    res = forward_last_logits(model, tokenizer, "hi", output_hidden_states=True)
    assert res.hidden_last is None


def test_forward_last_logits_with_mock_model():
    p = torch.nn.Parameter(torch.zeros(1, device="cpu"))
    model = MagicMock()
    model.parameters.return_value = iter([p])
    logits = torch.randn(1, 5, 32)
    hidden = torch.randn(1, 5, 16)
    out = MagicMock()
    out.logits = logits
    out.hidden_states = (torch.randn(1, 5, 8), hidden)
    model.return_value = out

    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    res = forward_last_logits(model, tokenizer, "hi")
    assert res.logits_last.shape == (32,)
    assert res.hidden_last is not None
    assert res.hidden_last.shape == (16,)


def test_hf_smoke_forward_injects_loader(monkeypatch):
    monkeypatch.setattr(hf, "transformers_available", lambda: True)

    def fake_loader(name, **kwargs):
        p = torch.nn.Parameter(torch.zeros(1))
        model = MagicMock()
        model.parameters.return_value = iter([p])
        model.to = MagicMock(return_value=model)
        logits = torch.randn(1, 3, 12)
        hlast = torch.randn(1, 3, 7)
        out = MagicMock()
        out.logits = logits
        out.hidden_states = (hlast,)
        model.return_value = out
        tok = MagicMock()
        tok.return_value = {"input_ids": torch.tensor([[1, 2]])}
        return model, tok

    res = hf_smoke_forward("dummy", "p", load_fn=fake_loader, device_map=None)
    assert res.logits_last.numel() == 12


def test_hf_smoke_report_prints(capsys, monkeypatch):
    monkeypatch.setattr(hf, "transformers_available", lambda: True)

    def fake_loader(name, **kwargs):
        p = torch.nn.Parameter(torch.zeros(1))
        model = MagicMock()
        model.parameters.return_value = iter([p])
        model.to = MagicMock(return_value=model)
        logits = torch.randn(1, 2, 5)
        out = MagicMock()
        out.logits = logits
        out.hidden_states = (torch.randn(1, 2, 5),)
        model.return_value = out
        tok = MagicMock()
        tok.return_value = {"input_ids": torch.tensor([[1]])}
        return model, tok

    hf_smoke_report("m", "p", load_fn=fake_loader)
    out = capsys.readouterr().out
    assert "last-token entropy" in out


def test_hf_smoke_without_transformers(monkeypatch):
    monkeypatch.setattr(hf, "transformers_available", lambda: False)
    with pytest.raises(RuntimeError, match="transformers"):
        hf_smoke_forward("x", "y", load_fn=lambda *a, **k: (None, None))


def test_load_causal_lm_mutually_exclusive_quant_raises(monkeypatch):
    pytest.importorskip("transformers")
    with pytest.raises(ValueError, match="at most one"):
        load_causal_lm("gpt2", load_in_4bit=True, load_in_8bit=True, device_map=None)
