#!/usr/bin/env python3
"""
Demo: fixed-budget loop + signals + bit-aware calibration + halting policy.

Usage (from repo root):
  set PYTHONPATH=src
  python scripts/run_baseline_demo.py
  python scripts/run_baseline_demo.py --config configs/default.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without install
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from bitcal_tts.demo import run_demo


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
