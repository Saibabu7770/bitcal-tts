from pathlib import Path

from bitcal_tts.config import load_config


def test_load_default_yaml():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    assert cfg["project"]["name"] == "bitcal-tts"
    assert "policy" in cfg
