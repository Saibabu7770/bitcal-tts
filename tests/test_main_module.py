import os
import runpy
import subprocess
import sys
from pathlib import Path


def test_runpy_main_invokes_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bitcal-tts", "doctor"])
    runpy.run_module("bitcal_tts.__main__", run_name="__main__")
    out = capsys.readouterr().out
    assert "BitCal-TTS" in out


def test_module_invocation_doctor():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "bitcal_tts", "doctor"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0
    assert "BitCal-TTS" in proc.stdout
