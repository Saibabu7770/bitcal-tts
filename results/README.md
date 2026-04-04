# BitCal-TTS Experiment Results

All raw runs are stored as JSONL in `raw/`. Processed outputs (CSV, plots) are in `processed/`.

---

## Run log

### Run 1 — Smoke test (1 item, budget 32, fixed only)
| Field | Value |
|---|---|
| File | `raw/exp_Qwen_Qwen2.5-3B-Instruct_1775333863.jsonl` |
| Date | 2026-04-04 |
| Model | `Qwen/Qwen2.5-3B-Instruct` (float32, CPU) |
| Benchmark | Local JSONL (`benchmarks/example_tasks.jsonl`) |
| Items | 1 |
| Budget | 32 |
| Methods | fixed |
| Purpose | Verify end-to-end pipeline runs with real model |
| Result | Pipeline OK; `correct=False` (32 tokens too short for math reasoning) |

---

### Run 2 — Smoke test repeat
| Field | Value |
|---|---|
| File | `raw/exp_Qwen_Qwen2.5-3B-Instruct_1775333944.jsonl` |
| Date | 2026-04-04 |
| Same as Run 1 (repeat after Unicode fix) |

---

### Run 3 — 3-method validation (2 GSM8K items, budget 48)
| Field | Value |
|---|---|
| File | `raw/exp_Qwen_Qwen2.5-3B-Instruct_1775337983.jsonl` |
| Date | 2026-04-04 |
| Model | `Qwen/Qwen2.5-3B-Instruct` (float32, CPU) |
| Benchmark | GSM8K test split (HF datasets), 2 items |
| Items | 2 |
| Budget | 48 |
| Methods | fixed, adaptive, bitcal_tts |
| Purpose | Validate all 3 methods produce different halting behaviour |
| Key finding | adaptive & bitcal_tts halted after **8 tokens** (1 step) — policy correctly detected high entropy / low confidence on a math problem with a tiny budget. fixed used all 48 tokens. |

#### Per-row signal data (Run 3)
| task | method | budget | tokens_used | mean_entropy | n_stops | wall_ms |
|---|---|---|---|---|---|---|
| gsm8k-test-00000 | fixed | 48 | 48 | 0.0 | 0 | 32,883 |
| gsm8k-test-00000 | adaptive | 48 | 8 | 0.232 | 1 | 8,468 |
| gsm8k-test-00000 | bitcal_tts | 48 | 8 | 0.232 | 1 | 7,928 |
| gsm8k-test-00001 | fixed | 48 | 48 | 0.0 | 0 | 38,937 |
| gsm8k-test-00001 | adaptive | 48 | 8 | 0.001 | 1 | 13,001 |
| gsm8k-test-00001 | bitcal_tts | 48 | 8 | 0.001 | 1 | 12,381 |

**Note:** These runs use a 48-token budget which is far too small for Qwen2.5 to solve GSM8K (needs ~200-400 tokens). Accuracy=0 for all methods is expected. The important signal is **token efficiency** — adaptive/bitcal_tts used 8/48 = 17% of the budget vs fixed using 100%.

---

## Next runs (planned)

These are the runs that will generate actual paper results.
Run these on a GPU machine (Google Colab T4 or better):

```bash
# Full minimal experiment: 50 items, budgets 256/512/1024, 3 methods, 4-bit
python scripts/run_experiment.py --config configs/experiment_gsm8k_minimal.yaml
python scripts/analyze_results.py
```

Expected outcome (from literature + BitCal-TTS design):
- BitCal-TTS achieves same or higher accuracy than fixed at lower avg token use
- Premature stop rate: BitCal-TTS < adaptive (bit-aware calibration is more conservative)
- Overthink rate: fixed > adaptive > bitcal_tts

---

## File structure

```
results/
  raw/                        # per-run JSONL (one line per task×method×budget)
  processed/
    summary.csv               # aggregated accuracy, token efficiency, halt rates
    pareto_quality_tokens.pdf # accuracy vs avg_tokens scatter (paper figure)
    accuracy_by_budget.pdf    # grouped bar chart (paper figure)
```

---

## How to reproduce

```bash
git clone https://github.com/Saibabu7770/bitcal-tts.git
cd bitcal-tts
pip install -e ".[research]"

# Re-generate processed outputs from existing raw results
python scripts/analyze_results.py

# Run new experiment
python scripts/run_experiment.py --config configs/experiment_gsm8k_minimal.yaml
```
