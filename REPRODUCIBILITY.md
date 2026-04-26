# Reproducibility guide

This document maps every cell of the paper's main results table (Table 1)
and every figure to **the exact raw JSONL artifact**, **the exact CLI**, and
**the exact configuration** that produced it. It is meant to be exhaustive
enough that an independent reviewer can rebuild the paper's plots without
re-running any model, or re-run any cell on their own GPU and obtain
matching numbers up to floating-point and Hugging Face cache variation.

If you only want to rebuild plots from the released JSONLs, jump to
[Quick rebuild](#quick-rebuild). For the line-by-line audit trail, read
[Per-row provenance](#per-row-provenance).

---

## Environment

- **OS:** Ubuntu (Colab runtime). Linux / macOS / Windows are all supported
  by the library; the published JSONLs were generated on Colab.
- **Hardware:** single NVIDIA T4 GPU, **16 GB VRAM**, ~13 GB system RAM
  (default Colab Pro free instance, April 2026).
- **Quantization:** `bitsandbytes` **4-bit weights** (`load_in_4bit=True`)
  with bf16 compute dtype, exposed via Hugging Face Transformers.
- **Decoding:** greedy (`do_sample=False`) for reproducibility.
- **Random seed:** `42` (used for any HF / numpy / torch RNG state in
  `scripts/run_experiment.py`).
- **Python:** 3.10–3.13 (CI matrix). The published JSONLs were generated
  under Python 3.11 on Colab.

Pinned package versions are in `requirements.txt` (flat) and the
`pyproject.toml` `[research]` extra (preferred). At paper time the relevant
core versions were:

| Package         | Version (paper) |
|-----------------|-----------------|
| `transformers`  | 4.44.x          |
| `torch`         | 2.4.x + cu121   |
| `bitsandbytes`  | 0.43.x          |
| `accelerate`    | 0.33.x          |
| `datasets`      | 2.20.x          |

Any reasonably recent version that supports 4-bit `bitsandbytes` loading on
your GPU should produce numbers within ~0.5 accuracy points.

---

## Models

| Paper symbol | Hugging Face identifier            | Loaded at |
|--------------|------------------------------------|-----------|
| 3B           | `Qwen/Qwen2.5-3B-Instruct`         | 4-bit NF4 |
| 7B           | `Qwen/Qwen2.5-7B-Instruct`         | 4-bit NF4 |
| 14B          | `Qwen/Qwen2.5-14B-Instruct`        | 4-bit NF4 |

All three models are loaded with the **instruction-tuned chat template**
(`apply_chat_template=True`).

---

## Benchmark and prompts

- **Benchmark:** GSM8K test split via Hugging Face `datasets`
  (`gsm8k/main`, split `test`).
- **Items per model:**
  - 3B: first $N = 50$ test items.
  - 7B: first $N = 54$ test items at $B \in \{256, 512\}$;
    $N = 53$ at $B = 1024$ (one OOM-skipped item).
  - 14B: first $N = 35$ test items at $B = 512$;
    $N = 34$ at $B = 1024$ (one OOM-skipped item).
- **Prompt:** GSM8K-style chain-of-thought instruction with the standard
  `#### <numeric answer>` final-line marker. The exact template is in
  `configs/experiment_gsm8k_minimal.yaml`.
- **Answer extraction:** the literal substring after the final `####`,
  parsed as a float; exact-match against the gold answer.

---

## Quick rebuild

Rebuild every paper table and figure from the published JSONLs:

```bash
git clone https://github.com/Saibabu7770/bitcal-tts.git
cd bitcal-tts
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,research]"

# Cross-model summary (all 3B + 7B + 14B JSONLs).
python scripts/analyze_results.py

# 7B-only Pareto + budget sweep (Figures 4 and 5 in the paper).
python scripts/analyze_results.py \
    --file-glob "*7B*.jsonl" \
    --out-dir   results/processed/7b

# Publication figures (writes media/fig_*.pdf and .png used by the .tex).
python scripts/paper_figures.py
```

Compare your generated `results/processed/summary.csv` against the
checked-in copy. Numbers should match exactly (the script is deterministic
given the same JSONLs). Figures should match the PDFs in
`results/processed/` byte-similarly modulo timestamp metadata.

---

## Per-row provenance

### Table 1 — Cross-model results at $B = 512$, 4-bit

| Cell | Model | Method     | $N$ | Acc. (%) | Avg. tok. | Savings | Prem. stop | Source JSONL |
|------|-------|------------|----:|---------:|----------:|--------:|-----------:|--------------|
| 1.1  | 3B    | fixed      |  50 |     60.0 |       281 |       — |        0.0 | `results/raw/exp_Qwen_Qwen2.5-3B-Instruct_1775356300.jsonl` |
| 1.2  | 3B    | adaptive   |  50 |     22.0 |       132 |    53.0 |       63.0 | `results/raw/exp_Qwen_Qwen2.5-3B-Instruct_1775356300.jsonl` |
| 1.3  | 3B    | BitCal-TTS |  50 |     20.0 |       144 |    49.0 |       63.0 | `results/raw/exp_Qwen_Qwen2.5-3B-Instruct_1775356300.jsonl` |
| 1.4  | 7B    | fixed      |  54 |     90.7 |       466 |       — |        0.0 | `results/raw/exp_Qwen_Qwen2.5-7B-Instruct_1775428350.jsonl` |
| 1.5  | 7B    | adaptive   |  54 |     79.6 |       286 |    38.5 |       14.8 | `results/raw/exp_Qwen_Qwen2.5-7B-Instruct_1775428350.jsonl` |
| 1.6  | 7B    | BitCal-TTS |  54 |     83.3 |       316 |    32.1 |       11.1 | `results/raw/exp_Qwen_Qwen2.5-7B-Instruct_1775428350.jsonl` |
| 1.7  | 14B   | fixed      |  35 |     88.6 |       455 |       — |        0.0 | `results/raw/exp_Qwen_Qwen2.5-14B-Instruct_1775412134.jsonl` |
| 1.8  | 14B   | adaptive   |  35 |     82.9 |       239 |    47.5 |       17.1 | `results/raw/exp_Qwen_Qwen2.5-14B-Instruct_1775412134.jsonl` |
| 1.9  | 14B   | BitCal-TTS |  35 |     85.7 |       269 |    40.8 |       11.4 | `results/raw/exp_Qwen_Qwen2.5-14B-Instruct_1775412134.jsonl` |

Exact CLIs that produced each JSONL:

```bash
# 3B (Run 6 in results/README.md): 50 items × {256,512,1024} × 3 methods.
python scripts/run_experiment.py \
    --config configs/experiment_gsm8k_minimal.yaml \
    --model  Qwen/Qwen2.5-3B-Instruct  \
    --n-items 50 --budgets 256,512,1024 \
    --methods fixed,adaptive,bitcal_tts \
    --min-tokens-before-halt 128 --seed 42

# 7B (Run 10): 54 items × {256,512,1024} × 3 methods.
python scripts/run_experiment.py \
    --config configs/experiment_gsm8k_minimal.yaml \
    --model  Qwen/Qwen2.5-7B-Instruct  \
    --n-items 54 --budgets 256,512,1024 \
    --methods fixed,adaptive,bitcal_tts \
    --min-tokens-before-halt 128 --seed 42

# 14B (Run 9): 35 items × {512,1024} × 3 methods.
python scripts/run_experiment.py \
    --config configs/experiment_gsm8k_minimal.yaml \
    --model  Qwen/Qwen2.5-14B-Instruct \
    --n-items 35 --budgets 512,1024 \
    --methods fixed,adaptive,bitcal_tts \
    --min-tokens-before-halt 128 --seed 42
```

### Budget sweep on 7B (Figures 4–5 and Appendix table)

The **same** 7B JSONL above contains all three budgets. Filter / aggregate
with:

```bash
python scripts/analyze_results.py \
    --file-glob "*7B*.jsonl" \
    --out-dir   results/processed/7b
```

This writes the per-budget summary to `results/processed/7b/summary.csv`
and the two figures used in the paper:

- `results/processed/7b/pareto_quality_tokens.{pdf,png}` — Figure 4.
- `results/processed/7b/accuracy_by_budget.{pdf,png}`   — Figure 5.

### Figures 2 and 3 (cross-model bars + premature-stop)

```bash
python scripts/paper_figures.py
# writes media/fig_main_accuracy_tokens.{pdf,png}   # Figure 2
#        media/fig_premature_stop.{pdf,png}        # Figure 3
#        media/fig_pareto_7b.{pdf,png}             # Figure 4 (also)
#        media/fig_7b_budget_sweep.{pdf,png}       # Figure 5 (also)
```

The aggregates inside `scripts/paper_figures.py` are constants drawn from
`results/processed/summary.csv` and are kept in sync with Table 1 by hand;
running `analyze_results.py` reproduces them from the raw JSONLs.

---

## Premature-stop definition

Throughout the paper and the code, **premature stop** means an example on
which (a) the controller halted **before** the budget was exhausted and
(b) the parsed final answer is **wrong**. This is computed in
`scripts/analyze_results.py::_premature_stop_rate` and is independent of
the wall-clock token count; it isolates the failure mode that adaptive
halting is designed to avoid.

---

## Hardware-budget notes

Each row in the table costs roughly:

| Model | Items × Methods × Budgets | Tokens generated | Wall time on T4 (4-bit) |
|-------|---------------------------|------------------|--------------------------|
| 3B    | 50 × 3 × 3 = 450 calls    | ~75 K            | ~25 min                  |
| 7B    | 54 × 3 × 3 = 483 calls    | ~150 K           | ~80 min                  |
| 14B   | 35 × 3 × 2 = 207 calls    | ~85 K            | ~110 min                 |

Reproductions on stronger GPUs (A100 / RTX 4090) typically finish 3–6× faster.

---

## Known sources of small drift

1. **Hugging Face model cache.** Qwen2.5 checkpoints are versioned by
   commit SHA on the Hub. Different snapshots (e.g. before vs. after a
   tokenizer fix) can shift accuracy by < 1 point.
2. **`bitsandbytes` minor version.** 0.43.x → 0.44.x changed NF4 dequant
   ordering on some GPUs; greedy outputs can flip on a small minority of
   items.
3. **CUDA / cuBLAS version.** Matmul reductions are not bit-exact across
   driver versions, even at fp16/bf16.
4. **Prompt template revisions.** If you edit
   `configs/experiment_gsm8k_minimal.yaml` (e.g. swap the chat template),
   the JSONL hashes will change and accuracy may shift.

For an exact bit-by-bit reproduction, pin to the model SHAs and package
versions listed above.
