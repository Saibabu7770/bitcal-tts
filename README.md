# BitCal-TTS

**Bit-Calibrated Test-Time Scaling for Quantized Reasoning Models**

[![CI](https://github.com/Saibabu7770/bitcal-tts/actions/workflows/ci.yml/badge.svg)](https://github.com/Saibabu7770/bitcal-tts/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Lightweight, **model-agnostic** control loop for **budgeted reasoning** with quantized LLMs: online uncertainty signals, **bit-aware** confidence calibration, and **continue / stop / escalate** decisions—without retraining the base model.

**Current release:** `v0.1.0` (research / alpha). Install from this repository; [PyPI publishing](https://pypi.org) is not set up yet—use `git clone` or `pip install` from GitHub (see below).

---

## For new users (quick checklist)

| Step | Command | What success looks like |
|------|---------|-------------------------|
| 1. Get the code | Clone or `pip install` from Git (below) | You have `bitcal_tts` importable |
| 2. Install deps | `pip install -e ".[dev,research]"` from repo root | No install errors |
| 3. Verify | `bitcal-tts doctor` | Prints Python, torch, transformers, PyYAML versions |
| 4. Run demo | `bitcal-tts demo --max-steps 2` | Prints steps, metrics, halting actions |
| 5. Run tests | `python -m pytest tests/ -q --no-cov` | `passed` (or use default pytest for coverage gate) |

If all five pass on your machine, your environment matches what we test in [CI](https://github.com/Saibabu7770/bitcal-tts/actions) (Ubuntu, Python 3.10–3.13).

---

## Requirements

- **Python** 3.10, 3.11, 3.12, or 3.13
- **OS:** Linux, macOS, or Windows
- **Hardware:** CPU is enough for tests and the mock demo; a **GPU** is optional for real model runs (`hf-smoke`, future experiments)
- **Disk / network:** optional Hugging Face commands download model weights on first use

---

## Installation

### Option A — Clone (recommended)

```bash
git clone https://github.com/Saibabu7770/bitcal-tts.git
cd bitcal-tts
python -m venv .venv
```

Activate the venv:

- **Linux / macOS:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **Windows (cmd):** `.\.venv\Scripts\activate.bat`

Then:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev,research]"
```

### Option B — Install from GitHub without cloning (pip)

Install the package directly (non-editable):

```bash
pip install "bitcal-tts[dev,research] @ git+https://github.com/Saibabu7770/bitcal-tts.git"
```

Minimal (library + tests only):

```bash
pip install "bitcal-tts[dev] @ git+https://github.com/Saibabu7770/bitcal-tts.git"
```

### Option C — Flat `requirements.txt`

From a clone:

```bash
pip install -r requirements.txt
```

For development you still should install the package in editable mode: `pip install -e ".[dev,research]"`.

---

## Quick start

**Mock demo** (no GPU, no model download):

```bash
python -m bitcal_tts demo
# or
bitcal-tts demo --config configs/default.yaml
```

**Environment check:**

```bash
bitcal-tts doctor
```

**Optional — Hugging Face smoke test** (downloads a small model; needs `[research]`):

```bash
bitcal-tts hf-smoke --model gpt2 --prompt "Hello"
```

On a machine with **CUDA** and a proper PyTorch build, `hf-smoke` can use the GPU automatically.

**Legacy entry point** (same as `demo`):

```bash
python scripts/run_baseline_demo.py --config configs/default.yaml
```

---

## Testing

Run the full suite **without** the coverage gate (fast):

```bash
python -m pytest tests/ -q --no-cov
```

Run with **coverage** (same as default `pytest` in this repo; enforces ≥90% line coverage on `bitcal_tts`):

```bash
python -m pytest tests/ -q
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
  .github/workflows/  # CI
```

---

## Configuration

Edit [`configs/default.yaml`](configs/default.yaml) for token budgets, policy thresholds, and calibrator settings (bit width, temperature). The demo merges CLI flags with YAML when `--config` is passed.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| **`git push` asks for password and fails** | GitHub requires a **Personal Access Token** (or SSH), not your account password. See [GitHub docs on HTTPS](https://docs.github.com/en/get-started/git-basics/about-remote-repositories). |
| **`transformers` / model download errors** | Install extras: `pip install -e ".[research]"`. Check network and disk space. |
| **`bitsandbytes` / 4-bit load fails** | Optional dependency; install per [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) for your OS/GPU. Not required for tests. |
| **Tests fail only with coverage** | Run `pytest tests/ --no-cov` first. If that passes, failures are coverage-related or environment-specific. |
| **CUDA not seen** | Install the PyTorch build that matches your CUDA from [pytorch.org](https://pytorch.org). `bitcal-tts doctor` shows `cuda available: True/False`. |

---

## Why BitCal-TTS?

Quantized reasoning models are efficient but **confidence signals** used for adaptive inference (entropy, trace stability, hidden-state agreement) can be miscalibrated relative to full precision. Under a **fixed token budget**, that hurts accuracy–efficiency tradeoffs.

**BitCal-TTS** adds **quantization-aware calibration** on top of standard signals so halting respects effective precision (e.g., 4-bit vs 8-bit vs 16-bit), in a reproducible open-source pipeline.

---

## Features

| Component | Description |
|-----------|-------------|
| **Signals** | Token entropy, reasoning-trace stability, optional hidden-state stability |
| **Calibration** | Bit-aware confidence mapping (more conservative at lower effective precision) |
| **Policy** | Halting: `continue`, `stop`, `escalate` |
| **Evaluation** | Trace summaries and halting metrics (tokens, escalations, efficiency) |
| **Integration** | Optional Hugging Face forward pass (`hf-smoke`) |

Core tests run on **CPU** with mocks; large-model experiments are optional.

---

## Roadmap

- **First paper milestone:** [docs/MINIMAL_EXPERIMENT.md](docs/MINIMAL_EXPERIMENT.md) — one model, one benchmark (e.g. GSM8K subset), three methods, ~8 GB VRAM
- Baseline sweeps on public reasoning benchmarks
- Optional **vLLM** / server-style integration
- Paper + artifact bundle when results meet [`PROJECT_PLAN.md`](PROJECT_PLAN.md) criteria

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Security

See [SECURITY.md](SECURITY.md).

---

## Citation

If you use this code in research:

```bibtex
@software{bitcal_tts2026,
  title        = {BitCal-TTS: Bit-Calibrated Test-Time Scaling for Quantized Reasoning Models},
  year         = {2026},
  url          = {https://github.com/Saibabu7770/bitcal-tts},
  note         = {Open-source research implementation}
}
```

---

## License

[MIT License](LICENSE)

---

## Copyright

Copyright (c) 2026, Sai Babu All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
