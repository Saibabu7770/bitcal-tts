#!/usr/bin/env python3
"""
BitCal-TTS full experiment runner.

Usage (from repo root with PYTHONPATH=src or after pip install -e .):

  # Minimal first experiment (Qwen2.5-3B, GSM8K 50, three methods):
  python scripts/run_experiment.py --config configs/experiment_gsm8k_minimal.yaml

  # Quick smoke run (2 items, CPU, tiny model):
  python scripts/run_experiment.py \\
      --model gpt2 --n-samples 2 --budget 64 --step-size 8 \\
      --no-4bit --source jsonl \\
      --jsonl-path benchmarks/example_tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from bitcal_tts.config import load_config
from bitcal_tts.experiment import METHODS, run_item
from bitcal_tts.integrations.hf_inference import load_causal_lm
from bitcal_tts.policy.halting import HaltingPolicy
from benchmarks.gsm8k_loader import answers_match, extract_answer, load_gsm8k


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="BitCal-TTS experiment runner")
    ap.add_argument("--config", type=str, default=None, help="YAML experiment config")
    ap.add_argument("--model", type=str, default=None, help="HF model id / local path")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--budget", type=int, nargs="+", default=[256, 512, 1024],
                    help="One or more token budgets")
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    choices=list(METHODS),
                    help="Methods to run")
    ap.add_argument("--bit-width", type=int, default=4,
                    help="Effective quantization bit width (for calibrator)")
    ap.add_argument("--no-4bit", action="store_true",
                    help="Disable 4-bit quantization (use float16)")
    ap.add_argument("--step-size", type=int, default=16,
                    help="Tokens per step in adaptive methods")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source", type=str, default="hf", choices=["hf", "jsonl"])
    ap.add_argument("--jsonl-path", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default="results/raw",
                    help="Directory for per-run JSONL logs")
    ap.add_argument("--max-entropy", type=float, default=10.0)
    ap.add_argument("--stop-entropy", type=float, default=2.0)
    ap.add_argument("--escalate-entropy", type=float, default=5.0)
    ap.add_argument("--stop-conf", type=float, default=0.75)
    ap.add_argument("--min-budget-continue", type=int, default=16)
    ap.add_argument("--min-tokens-before-halt", type=int, default=128,
                    help="Minimum tokens generated before halting policy is allowed to stop")
    return ap


def set_seed(seed: int) -> None:
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config into args (CLI flags override config)."""
    if args.config:
        cfg = load_config(args.config)
        m = cfg.get("model", {})
        exp = cfg.get("experiment", {})
        p = cfg.get("policy", {})
        c = cfg.get("calibrator", {})
        ev = cfg.get("eval", {})
        proj = cfg.get("project", {})

        args.model = args.model or m.get("name_or_path") or None
        args.n_samples = exp.get("n_samples", args.n_samples)
        args.seed = proj.get("seed", args.seed)
        if "budgets" in exp:
            args.budget = exp["budgets"]
        args.methods = exp.get("methods", args.methods)
        args.step_size = exp.get("step_size", args.step_size)
        args.source = exp.get("benchmark_source", args.source)
        args.bit_width = c.get("bit_width_default", args.bit_width)
        args.stop_entropy = p.get("stop_entropy_threshold", args.stop_entropy)
        args.escalate_entropy = p.get("escalate_entropy_threshold", args.escalate_entropy)
        args.stop_conf = p.get("stop_confidence_threshold", args.stop_conf)
        args.min_budget_continue = p.get("min_budget_to_continue", args.min_budget_continue)
        args.min_tokens_before_halt = cfg.get("min_tokens_before_halt", args.min_tokens_before_halt)
        args.output_dir = ev.get("output_dir", args.output_dir)
        load_in_4bit = m.get("load_in_4bit", not args.no_4bit)
        args.no_4bit = not load_in_4bit

    return args


def save_results(results: List[Dict[str, Any]], out_dir: Path, run_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return out_path


def open_streaming_file(out_dir: Path, run_id: str) -> tuple:
    """Open a JSONL file for streaming (one row written per result immediately)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    fh = out_path.open("w", encoding="utf-8")
    return fh, out_path


def print_summary(results: List[Dict[str, Any]]) -> None:
    from collections import defaultdict

    buckets: dict = defaultdict(list)
    for r in results:
        key = (r["method"], r["budget"])
        buckets[key].append(r)

    print(f"\n{'Method':<14} {'Budget':>7}  {'N':>4}  {'Acc':>6}  {'AvgTok':>8}  "
          f"{'Stops':>6}  {'Escalate':>8}")
    print("-" * 65)
    for (meth, bud), rows in sorted(buckets.items()):
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / n if n else 0.0
        avg_tok = sum(r["tokens_used"] for r in rows) / n if n else 0.0
        n_stops = sum(r["n_stops"] for r in rows)
        n_esc = sum(r["n_escalations"] for r in rows)
        print(f"{meth:<14} {bud:>7}  {n:>4}  {acc:>6.3f}  {avg_tok:>8.1f}  "
              f"{n_stops:>6}  {n_esc:>8}")
    print()


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args = resolve_args(args)

    if not args.model:
        print("[ERROR] --model is required (e.g. Qwen/Qwen2.5-3B-Instruct)")
        sys.exit(1)

    set_seed(args.seed)

    print(f"Loading dataset (source={args.source}, n={args.n_samples})...")
    jsonl_path = args.jsonl_path or str(_ROOT / "benchmarks" / "example_tasks.jsonl")
    tasks = load_gsm8k(
        source=args.source,
        n_samples=args.n_samples,
        seed=args.seed,
        jsonl_path=jsonl_path if args.source == "jsonl" else None,
    )
    print(f"  Loaded {len(tasks)} items.")

    print(f"Loading model: {args.model}  (4-bit={not args.no_4bit}) ...")
    model, tokenizer = load_causal_lm(
        args.model,
        load_in_4bit=not args.no_4bit,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to(torch.device("cpu"))
    model.eval()
    print("  Model loaded.")

    policy = HaltingPolicy(
        stop_entropy_threshold=args.stop_entropy,
        escalate_entropy_threshold=args.escalate_entropy,
        stop_confidence_threshold=args.stop_conf,
        min_budget_to_continue=args.min_budget_continue,
    )

    run_id = f"exp_{args.model.replace('/', '_')}_{int(time.time())}"
    out_dir = _ROOT / args.output_dir

    all_results: List[Dict[str, Any]] = []
    total = len(tasks) * len(args.budget) * len(args.methods)
    done = 0

    # Stream each result to disk immediately so partial runs are never lost.
    stream_fh, out_path = open_streaming_file(out_dir, run_id)
    print(f"Streaming results -> {out_path}")

    try:
        for task in tasks:
            for bud in args.budget:
                for method in args.methods:
                    done += 1
                    print(f"[{done}/{total}] task={task.id}  method={method}  budget={bud}", end=" ")
                    try:
                        result = run_item(
                            model=model,
                            tokenizer=tokenizer,
                            task_id=task.id,
                            prompt=task.prompt,
                            gold_answer=task.gold_answer,
                            method=method,
                            budget=bud,
                            bit_width=args.bit_width,
                            policy=policy,
                            seed=args.seed,
                            step_size=args.step_size,
                            max_entropy=args.max_entropy,
                            answer_extractor=extract_answer,
                            min_tokens_before_halt=args.min_tokens_before_halt,
                        )
                        r = result.to_dict()
                        print(f"-> correct={result.correct}  tokens={result.tokens_used}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"-> ERROR: {exc}")
                        r = {
                            "task_id": task.id, "method": method, "budget": bud,
                            "error": str(exc), "correct": False, "tokens_used": 0,
                        }
                    all_results.append(r)
                    stream_fh.write(json.dumps(r) + "\n")
                    stream_fh.flush()
    finally:
        stream_fh.close()

    print(f"\nSaved {len(all_results)} results -> {out_path}")
    print_summary(all_results)


if __name__ == "__main__":
    main()
