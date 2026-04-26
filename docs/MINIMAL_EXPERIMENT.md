# Minimal first experiment (one page of results)

> **Status: completed and superseded (April 2026).** The minimal protocol
> below was the original v0.1 plan (one model, one benchmark, ~50 items).
> It has since been **executed and extended** to all three Qwen2.5 sizes
> (3B / 7B / 14B) under 4-bit quantization on GSM8K, and the resulting
> tables and figures are the basis of the BitCal-TTS paper.
>
> For the **finished** results, see:
>
> - [`../results/README.md`](../results/README.md) — full run log with
>   per-run JSONL filenames, sample sizes, and the cross-model summary.
> - [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) — exact CLI for every
>   row of Table 1 and every figure of the paper.
> - [`../README.md`](../README.md) — headline numbers and quick rebuild.
>
> The text below is preserved verbatim as a historical record of the
> minimal-experiment scoping decision.

---

This narrows the full matrix in [PROJECT_PLAN.md](../PROJECT_PLAN.md) to **one GPU class**, **one model**, and **one benchmark** so you can get a first credible table/figure before scaling up.

## Assumed hardware (your profile)

- **GPU:** NVIDIA with **~8 GB VRAM** (e.g. RTX 4070 Laptop)
- **RAM:** 16+ GB system RAM (64 GB is plenty)
- **Implication:** Use **4-bit weight quantization** for models in the **3B–7B** range; avoid long contexts and huge batches.

If you have **12+ GB VRAM**, you can bump model size or context; if **only CPU**, use a **tiny** model (e.g. &lt;1B) for pipeline debugging—not for final paper numbers.

---

## One target model (recommended default)

| Option | Hugging Face id | Why |
|--------|-----------------|-----|
| **A (safest on 8 GB)** | `Qwen/Qwen2.5-3B-Instruct` | Fits comfortably in **4-bit**; strong enough for math-style reasoning; widely cited. |
| **B (stronger, tighter)** | `Qwen/Qwen2.5-7B-Instruct` | Better quality; **requires 4-bit** and modest `max_new_tokens` on 8 GB; watch OOM. |

**Quantization:** `load_in_4bit=True` (bitsandbytes) + bf16/fp16 compute as supported.

**Not chosen yet?** Pick **Option A** first; repeat the same protocol on Option B once the pipeline is stable.

---

## One benchmark (recommended default)

| Benchmark | What it is | Minimal subset |
|-----------|------------|----------------|
| **GSM8K** | Grade-school math word problems; standard for reasoning | Use **50–100** problems (fixed split: e.g. first **50** of `test` or a published subset) and **exact-match** on extracted final numeric answer |

**Why GSM8K:** One number to score, clear budget story (reasoning tokens vs accuracy), easy to explain in a paper.

**Alternatives** (if you already have tooling): MATH **level 1–2 only** (small subset), or a **custom JSONL** of 50 math problems with gold answers (same format as [benchmarks/example_tasks.jsonl](../benchmarks/example_tasks.jsonl)).

---

## Conditions to hold constant

- **Seeds:** one fixed seed for sampling (if any) and for PyTorch/HF.
- **Prompt:** Same instruction template for all methods (e.g. “Show short reasoning, then final answer as `#### <number>`” or your chosen CoT format).
- **Budgets:** Pick **three** token caps for reasoning+answer, e.g. **256 / 512 / 1024** total new tokens (tune down if OOM).
- **Quantization:** Same bit settings for all rows (e.g. **4-bit weights** only for v1).

---

## Methods (rows in your first table)

1. **Fixed budget** — Always generate until budget; no adaptive halting.
2. **Adaptive halting, not bit-aware** — Same policy thresholds, but **disable** bit-width scaling in the calibrator (or set a single effective scale).
3. **BitCal-TTS** — Full pipeline: signals + bit-aware calibration + policy.

---

## Metrics (columns worth one page)

- **Accuracy** (exact match on GSM8K subset)
- **Avg reasoning tokens used** (or total tokens)
- **Premature-stop / over-think rates** (if you log halting decisions)
- One **Pareto-style** point: accuracy vs tokens at each budget tier

---

## What you still implement in code

The repo has **signals, calibrator, policy, and mocks**. For this experiment you still need a **real HF loop** that:

1. Loads the quantized model.
2. Runs each GSM8K item with **logged per-step** logits / trace (as your design requires).
3. Writes **CSV/JSONL** under `results/raw/` with config hash + seed.

Use [configs/default.yaml](../configs/default.yaml) as a starting point and **snapshot** the exact YAML in the appendix.

---

## Suggested one-line decision record

> **Minimal v1:** `Qwen2.5-3B-Instruct` 4-bit, GSM8K **N=50**, budgets **256/512/1024**, three methods (fixed / adaptive / BitCal-TTS), fixed seed **42**.

Change only one knob at a time when moving to the “full matrix.”
