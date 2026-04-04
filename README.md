# BitCal-TTS

**Bit-Calibrated Test-Time Scaling for Quantized Reasoning Models**

[![CI](https://github.com/YOUR_ORG/bitcal-tts/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/bitcal-tts/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Lightweight, **model-agnostic** control loop for **budgeted reasoning** with quantized LLMs: online uncertainty signals, **bit-aware** confidence calibration, and **continue / stop / escalate** decisions—without retraining the base model.

> Replace `YOUR_ORG` in badge URLs and in `pyproject.toml` `[project.urls]` after you create the GitHub repository.

---

## Why BitCal-TTS?

Quantized reasoning models are efficient but **confidence signals used for adaptive inference** (entropy, trace stability, hidden-state agreement) can be miscalibrated relative to full precision. Under a **fixed token budget**, that leads to poor accuracy–efficiency tradeoffs: stopping too early, running too long, or inconsistent halting across bit widths.

**BitCal-TTS** applies a **quantization-aware calibration** layer on top of standard signals so halting decisions respect effective precision (e.g., 4-bit vs 8-bit vs 16-bit), improving **budgeted** reasoning quality in a reproducible, open-source pipeline.

---

## Features

| Component | Description |
|-----------|-------------|
| **Signals** | Token entropy, reasoning-trace stability, optional hidden-state stability |
| **Calibration** | Bit-aware confidence mapping (conservative at lower effective precision) |
| **Policy** | Halting actions: `continue`, `stop`, `escalate` (hook for more compute / precision) |
| **Evaluation** | Trace summaries and halting-centric metrics (tokens, escalations, efficiency) |
| **Integration** | Optional Hugging Face causal LM forward pass (`hf-smoke`) for development |

This repository is **research-oriented**: core tests run on **CPU** with mocks; real model runs are optional and use your GPU / cluster.

---

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/YOUR_ORG/bitcal-tts.git
cd bitcal-tts
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,research]"
```

### Minimal install (library + tests only)

```bash
pip install -e ".[dev]"
```

Optional: `pip install -r requirements.txt` for a flat dependency list aligned with experiments.

---

## Quick start

**Mock end-to-end demo** (no GPU, no downloads):

```bash
python -m bitcal_tts demo
# or
bitcal-tts demo --config configs/default.yaml
```

**Check environment**:

```bash
bitcal-tts doctor
```

**Optional Hugging Face smoke test** (downloads weights; requires `[research]`):

```bash
bitcal-tts hf-smoke --model gpt2 --prompt "Hello"
```

**Legacy script** (same as `demo`):

```bash
python scripts/run_baseline_demo.py --config configs/default.yaml
```

---

## Project layout

```text
bitcal-tts/
  src/bitcal_tts/     # Package: runner, signals, calibrator, policy, eval, integrations, CLI
  configs/            # YAML experiment templates
  benchmarks/         # JSONL task loader + example tasks
  scripts/            # Convenience runners
  tests/              # Pytest suite (CPU-safe)
  results/            # Local experiment outputs (gitignored except .gitkeep)
  .github/workflows/  # CI (pytest on push/PR)
```

---

## Configuration

Edit `configs/default.yaml` for token budgets, policy thresholds, and calibrator settings (bit width, temperature). The demo script merges CLI flags with YAML when `--config` is passed.

---

## Testing

```bash
python -m pytest tests/ -q
```

With coverage:

```bash
python -m pytest tests/ --cov=bitcal_tts --cov-report=term-missing
```

---

## Roadmap

- [ ] Full **baseline** sweeps: fixed budget vs BitCal-TTS vs non-bit-aware halting on public reasoning benchmarks
- [ ] **vLLM** / server-style integration for latency-aware evaluation
- [ ] **Paper** draft and artifact bundle (configs + logs + plots) after results meet success criteria in `PROJECT_PLAN.md`

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Citation

If you use this code in research, please cite (update when the paper is public):

```bibtex
@software{bitcal_tts2026,
  title        = {BitCal-TTS: Bit-Calibrated Test-Time Scaling for Quantized Reasoning Models},
  year         = {2026},
  url          = {https://github.com/YOUR_ORG/bitcal-tts},
  note         = {Open-source research implementation}
}
```

---

## License

[MIT License](LICENSE)

---

## Disclaimer

This is **research software**. It is not warranted for production deployment without your own safety, evaluation, and compliance review.
