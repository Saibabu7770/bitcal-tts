# BitCal-TTS

**Bit-Calibrated Test-Time Scaling for Quantized Reasoning Models**

<img width="848" height="485" alt="image" src="https://github.com/user-attachments/assets/a7de5452-5852-4107-b228-c0b07d1d6384" />

[![CI](https://github.com/Saibabu7770/bitcal-tts/actions/workflows/ci.yml/badge.svg)](https://github.com/Saibabu7770/bitcal-tts/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/bitcal-tts.svg)](https://pypi.org/project/bitcal-tts/)
[![Paper](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2026.XXXXX)


> *BitCal-TTS: Bit-Calibrated Test-Time Scaling for Quantized Reasoning Models*
> (preprint, April 2026).  `2026.XXXXX` above with the arXiv 
> 

Lightweight, **model-agnostic** runtime controller for **budgeted reasoning**
under post-training quantization: online uncertainty signals, **bit-aware**
confidence calibration, and **continue / stop / escalate** halting decisions —
**without retraining the base model**.



---

## Headline result (GSM8K, 4-bit, $B = 512$)

The numbers below come from `results/README.md` (Run 5/9/10) and exactly match
Table 1 . Raw per-task records are in
[`results/raw/`](results/raw); the same protocol can be re-run from this repo

| Model | Method     | Accuracy | Avg. tokens | Savings vs. fixed | Premature-stop |
|-------|------------|---------:|------------:|------------------:|---------------:|
| 3B    | fixed      |   60.0 % |         281 |                 — |          0.0 % |
| 3B    | adaptive   |   22.0 % |         132 |            53.0 % |         63.0 % |
| 3B    | BitCal-TTS |   20.0 % |         144 |            49.0 % |         63.0 % |
| **7B**  | **fixed**      | **90.7 %** |     **466** |               **—** |        **0.0 %** |
| **7B**  | **adaptive**   |   79.6 % |         286 |            38.5 % |         14.8 % |
| **7B**  | **BitCal-TTS** | **83.3 %** |         316 |            32.1 % |     **11.1 %** |
| **14B** | **fixed**      | **88.6 %** |     **455** |               **—** |        **0.0 %** |
| **14B** | **adaptive**   |   82.9 % |         239 |            47.5 % |         17.1 % |
| **14B** | **BitCal-TTS** | **85.7 %** |         269 |            40.8 % |     **11.4 %** |

Sample sizes: $N = 50$ (3B), $N = 54$ (7B), $N = 35$ (14B); all rows use
greedy decoding under 4-bit `bitsandbytes` weights on a single NVIDIA T4
(16 GB) Colab GPU.

**Takeaway.** BitCal-TTS adds **+3.7 accuracy points on Qwen2.5-7B** and
**+2.8 points on Qwen2.5-14B** over the precision-agnostic adaptive baseline,
while **roughly halving** its premature-stop rate, at a small extra token cost.

The companion budget sweep on 7B ($B \in \{256, 512, 1024\}$) and per-model
Pareto plots are in [`results/processed/`](results/processed) and
[`results/processed/7b/`](results/processed/7b).

---

## What is in this repository?

| Directory | Purpose |
|-----------|---------|
| `src/bitcal_tts/` | Library: signals, bit-aware calibrator, halting policy, runner, eval, HF integration, CLI. |
| `scripts/run_experiment.py` | End-to-end GSM8K runner used to produce the rows above. |
| `scripts/analyze_results.py` | Re-aggregates `results/raw/*.jsonl` into `results/processed/{summary.csv,*.pdf}`. |
| `scripts/paper_figures.py` | Generates the four publication figures (`media/fig_*.pdf`). |
| `configs/` | YAML experiment templates (`experiment_gsm8k_minimal.yaml`, `default.yaml`, …). |
| `benchmarks/` | JSONL task loader + tiny example task file for unit tests. |
| `tests/` | CPU-safe pytest suite (≥90 % line coverage on `bitcal_tts`). |
| `results/raw/` | Per-run JSONL traces (one line per task × method × budget) — . |
| `results/processed/` | Aggregated CSV + Pareto / accuracy-vs-budget figures. |
| `results/README.md` | Full run log: protocol, hardware, budget sweeps, cross-model summary. |
| `docs/` | Project plan, minimal-experiment notes, releasing notes. |

---

## Reproducing Results

The repository ships **the raw JSONL data** behind every  number, plus
the aggregation and plotting scripts, so you can rebuild every table and
figure offline without re-running any model.

### A. Rebuild the tables and figures from the released JSONLs

```bash
git clone https://github.com/Saibabu7770/bitcal-tts.git
cd bitcal-tts
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,research]"

# Cross-model summary (3B + 7B + 14B) used for the headline table:
python scripts/analyze_results.py

# 7B-only Pareto + budget sweep :
python scripts/analyze_results.py \
    --file-glob "*7B*.jsonl" \
    --out-dir results/processed/7b

# Publication figures (writes media/fig_*.pdf and .png):
python scripts/paper_figures.py
```

The expected outputs are already checked in under
[`results/processed/`](results/processed) so you can diff against them.

### B. Re-run the GSM8K experiments end-to-end (GPU required)

This reproduces the JSONLs in `results/raw/`. Each run takes 1–4 hours per
model on a Colab T4 (16 GB) under 4-bit quantization.

```bash
# 3B / 7B / 14B Qwen2.5-Instruct, GSM8K test split, fixed seed 42:
python scripts/run_experiment.py --config configs/experiment_gsm8k_minimal.yaml \
    --model Qwen/Qwen2.5-3B-Instruct  --n-items 50 --budgets 256,512,1024 \
    --methods fixed,adaptive,bitcal_tts --min-tokens-before-halt 128
python scripts/run_experiment.py --config configs/experiment_gsm8k_minimal.yaml \
    --model Qwen/Qwen2.5-7B-Instruct  --n-items 54 --budgets 256,512,1024 \
    --methods fixed,adaptive,bitcal_tts --min-tokens-before-halt 128
python scripts/run_experiment.py --config configs/experiment_gsm8k_minimal.yaml \
    --model Qwen/Qwen2.5-14B-Instruct --n-items 35 --budgets   512,1024 \
    --methods fixed,adaptive,bitcal_tts --min-tokens-before-halt 128
```

Or open [`colab_experiment.ipynb`](colab_experiment.ipynb) for the
ready-to-run Colab pipeline that produced the published JSONLs.

For an exhaustive row-by-row mapping (table cell → raw JSONL filename →
exact CLI), see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## Quickstart (no GPU required)

CPU-only smoke test of the controller logic on mock signals:

```bash
pip install -e ".[dev]"
bitcal-tts doctor                           # versions / CUDA visibility
bitcal-tts demo --config configs/default.yaml
python -m pytest tests/ -q
```

If a real GPU is present, you can sanity-check the Hugging Face integration:

```bash
pip install -e ".[research]"
bitcal-tts hf-smoke --model Qwen/Qwen2.5-3B-Instruct --prompt "1+1=" --quant 4bit
```

---

## Installation

### From PyPI

```bash
pip install "bitcal-tts[research]"      # full stack (transformers, bitsandbytes-friendly)
pip install bitcal-tts                  # library only
```

### From source

```bash
git clone https://github.com/Saibabu7770/bitcal-tts.git
cd bitcal-tts
python -m venv .venv
source .venv/bin/activate                # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev,research]"
```

### Without cloning

```bash
pip install "bitcal-tts[dev,research] @ git+https://github.com/Saibabu7770/bitcal-tts.git"
```

**Requirements.** Python 3.10–3.13 on Linux / macOS / Windows. CPU is
sufficient for tests and the mock demo. A CUDA GPU + a matching PyTorch build
is required for the GSM8K experiments; 4-bit `bitsandbytes` is recommended
(any model that fits in 8 GB VRAM at 4-bit will reproduce the 3B row).

---

## Method in 30 seconds

BitCal-TTS sits as a **sidecar around an unmodified quantized backbone**:

1. **Online signals.** Per chunk of $k = 16$ tokens, compute Shannon entropy
   $H_t$ on the last-position logits, a reasoning-trace stability score
   $\tau^{\mathrm{tr}}_t$ over recent decoded chunks, and (optionally) a
   hidden-state stability score $\tau^{\mathrm{hid}}_t$ via forward hooks.
2. **Bit-conditioned confidence.** Combine the signals into a raw confidence
   $c^{\mathrm{raw}}_t$, then rescale by a precision-aware factor
   $s(b) \in \{0.85, 1.00, 1.05\}$ for $b \le 4$, $4 < b \le 8$, $b > 8$.
3. **Halting policy.** A threshold rule on $(H_t, c_t)$ produces
   `continue / stop / escalate`. After the GSM8K answer marker `####` first
   appears, a precision-dependent confirmation tail $\Delta(b)$ (32 / 16 / 0
   tokens) suppresses premature stops driven by brittle low-bit formatting.

The only assumptions on the backbone are (a) per-step access to logits and
(b) optional access to last-layer hidden states. Both are exposed by standard
Hugging Face Transformers + `bitsandbytes` 4-bit inference.

See `arxiv:2026.XXXXX` Section 3 for full notation, equations, and the
algorithm pseudocode.

---

## Testing

```bash
python -m pytest tests/ -q --no-cov          # fast sanity run
python -m pytest tests/ -q                   # default; enforces ≥90% line coverage
```

CI runs the same suite on Ubuntu, Python 3.10–3.13.

---

## Citation

If you use BitCal-TTS in a paper, please cite the **paper** and (optionally) the **software**:

```bibtex
@misc{bitcal_tts_paper_2026,
  title         = {BitCal-TTS: Bit-Calibrated Test-Time Scaling for Quantized Reasoning Models},
  author        = {Sai Babu},
  year          = {2026},
  eprint        = {2026.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2026.XXXXX}
}

@software{bitcal_tts_software_2026,
  title  = {BitCal-TTS (software)},
  author = {Sai Babu},
  year   = {2026},
  url    = {https://github.com/Saibabu7770/bitcal-tts},
  note   = {Companion code for the BitCal-TTS arXiv preprint}
}
```

Replace `2026.XXXXX` with the assigned arXiv identifier once the preprint
goes live.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `git push` asks for a password | GitHub requires a Personal Access Token (or SSH), not your account password. See [GitHub docs](https://docs.github.com/en/get-started/git-basics/about-remote-repositories). |
| `transformers` / model download errors | Install the research extras: `pip install -e ".[research]"`. Check disk space and Hugging Face access. |
| `bitsandbytes` / 4-bit load fails | Install per [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) for your OS / CUDA. Not required for CPU tests. |
| Tests fail only with coverage | Run `pytest tests/ --no-cov` first to isolate environment issues from coverage gating. |
| CUDA not seen | Install the PyTorch build matching your CUDA from [pytorch.org](https://pytorch.org); `bitcal-tts doctor` reports `cuda available`. |

---

## Contributing & security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

---

## License

[MIT License](LICENSE) © 2026 Sai Babu. See `LICENSE` for the full text.
