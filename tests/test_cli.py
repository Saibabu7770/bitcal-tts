import sys
from unittest.mock import patch

import pytest

from bitcal_tts.cli import _print_doctor, main


def test_cli_demo_runs(capsys):
    main(["demo", "--max-steps", "1", "--budget", "64"])
    out = capsys.readouterr().out
    assert "BitCal-TTS baseline demo" in out


def test_cli_doctor(capsys):
    main(["doctor"])
    out = capsys.readouterr().out
    assert "BitCal-TTS" in out


def test_cli_default_runs_demo(capsys):
    main([])
    out = capsys.readouterr().out
    assert "BitCal-TTS baseline demo" in out


def test_cli_version_exits(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["--version"])
    assert ei.value.code == 0


def test_cli_hf_smoke_mocked(monkeypatch, capsys):
    from bitcal_tts.integrations import hf_inference as hf

    monkeypatch.setattr(hf, "transformers_available", lambda: True)

    def fake_report(model, prompt, load_fn=None):
        print("mock-smoke-ok")

    monkeypatch.setattr(hf, "hf_smoke_report", fake_report)
    main(["hf-smoke", "--model", "x", "--prompt", "y"])
    assert "mock-smoke-ok" in capsys.readouterr().out


def test_doctor_missing_torch(capsys):
    """Simulate torch not installed path in _print_doctor."""
    with patch.dict(sys.modules, {"torch": None}):
        _print_doctor()
    out = capsys.readouterr().out
    assert "torch" in out


def test_doctor_missing_transformers(capsys):
    """Simulate transformers not installed path in _print_doctor."""
    with patch.dict(sys.modules, {"transformers": None}):
        _print_doctor()
    out = capsys.readouterr().out
    assert "BitCal-TTS" in out


def test_doctor_missing_pyyaml(capsys):
    """Simulate pyyaml not installed path in _print_doctor."""
    with patch.dict(sys.modules, {"yaml": None}):
        _print_doctor()
    out = capsys.readouterr().out
    assert "BitCal-TTS" in out
