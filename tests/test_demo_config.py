from pathlib import Path

from bitcal_tts.demo import run_demo


def test_demo_with_config_file(tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
budget:
  max_reasoning_tokens: 64
policy:
  stop_entropy_threshold: 1.0
  escalate_entropy_threshold: 9.0
  min_budget_to_continue: 4
calibrator:
  bit_width_default: 8
  temperature: 1.0
""",
        encoding="utf-8",
    )
    run_demo(["--config", str(cfg), "--max-steps", "2", "--budget", "128"])
    out = capsys.readouterr().out
    assert "BitCal-TTS baseline demo" in out
    assert "metrics:" in out
