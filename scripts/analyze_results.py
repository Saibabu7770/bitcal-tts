#!/usr/bin/env python3
"""
Analyze BitCal-TTS experiment results.

Reads JSONL files from results/raw/, produces:
  - Summary table (accuracy, avg_tokens, premature_stop_rate, overthink_rate)
  - Pareto plot: accuracy vs avg_tokens (one point per method×budget)
  - Per-method accuracy-at-each-budget bar chart

Usage:
  python scripts/analyze_results.py --results-dir results/raw
  python scripts/analyze_results.py --results-dir results/raw --out-dir results/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def load_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(results_dir.glob("*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def compute_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate rows by (method, budget)."""
    buckets: dict = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        key = (r.get("method", "?"), int(r.get("budget", 0)))
        buckets[key].append(r)

    summary = []
    for (method, budget), items in sorted(buckets.items()):
        n = len(items)
        acc = sum(r.get("correct", False) for r in items) / n
        avg_tok = sum(r.get("tokens_used", 0) for r in items) / n

        # Premature stop: halted early AND wrong
        premature = sum(
            1 for r in items
            if r.get("n_stops", 0) > 0 and not r.get("correct", False)
            and r.get("tokens_used", budget) < budget
        )
        # Overthink: used full budget AND still wrong
        overthink = sum(
            1 for r in items
            if r.get("tokens_used", 0) >= budget and not r.get("correct", False)
        )

        summary.append({
            "method": method,
            "budget": budget,
            "n": n,
            "accuracy": round(acc, 4),
            "avg_tokens": round(avg_tok, 1),
            "token_efficiency": round(avg_tok / max(budget, 1), 4),
            "premature_stop_rate": round(premature / n, 4),
            "overthink_rate": round(overthink / n, 4),
        })
    return summary


def print_table(summary: List[Dict[str, Any]]) -> None:
    print(f"\n{'Method':<14} {'Budget':>7}  {'N':>4}  {'Accuracy':>9}  "
          f"{'AvgToks':>8}  {'TokEff':>7}  {'PremStop':>9}  {'Overthink':>10}")
    print("-" * 80)
    for row in summary:
        print(
            f"{row['method']:<14} {row['budget']:>7}  {row['n']:>4}  "
            f"{row['accuracy']:>9.4f}  {row['avg_tokens']:>8.1f}  "
            f"{row['token_efficiency']:>7.4f}  {row['premature_stop_rate']:>9.4f}  "
            f"{row['overthink_rate']:>10.4f}"
        )
    print()


def save_summary_csv(summary: List[Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "summary.csv"
    if not summary:
        return p
    keys = list(summary[0].keys())
    with p.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in summary:
            f.write(",".join(str(row[k]) for k in keys) + "\n")
    return p


def plot_pareto(
    summary: List[Dict[str, Any]],
    out_dir: Path,
) -> Optional[Path]:
    """Accuracy vs avg_tokens scatter with method labels."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping Pareto plot (pip install matplotlib)")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    method_color = {"fixed": "steelblue", "adaptive": "darkorange", "bitcal_tts": "green"}
    method_marker = {"fixed": "o", "adaptive": "s", "bitcal_tts": "^"}

    plotted = set()
    for row in summary:
        m = row["method"]
        label = m if m not in plotted else None
        plotted.add(m)
        ax.scatter(
            row["avg_tokens"],
            row["accuracy"],
            c=method_color.get(m, "gray"),
            marker=method_marker.get(m, "o"),
            s=80,
            label=label,
            alpha=0.85,
        )
        ax.annotate(
            str(row["budget"]),
            (row["avg_tokens"], row["accuracy"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=8,
        )

    ax.set_xlabel("Avg tokens used")
    ax.set_ylabel("Accuracy")
    ax.set_title("BitCal-TTS: Quality–Efficiency Pareto (GSM8K)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "pareto_quality_tokens.pdf"
    fig.savefig(out_path, dpi=150)
    out_path_png = out_dir / "pareto_quality_tokens.png"
    fig.savefig(out_path_png, dpi=150)
    plt.close(fig)
    return out_path


def plot_bar(
    summary: List[Dict[str, Any]],
    out_dir: Path,
) -> Optional[Path]:
    """Grouped bar chart: accuracy per method, per budget."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    budgets = sorted({r["budget"] for r in summary})
    methods = sorted({r["method"] for r in summary})
    acc_map: Dict[Tuple, float] = {(r["method"], r["budget"]): r["accuracy"] for r in summary}

    x = np.arange(len(budgets))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(methods):
        vals = [acc_map.get((m, b), 0.0) for b in budgets]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=m, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Token budget")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by method and token budget")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "accuracy_by_budget.pdf"
    fig.savefig(out_path, dpi=150)
    out_path_png = out_dir / "accuracy_by_budget.png"
    fig.savefig(out_path_png, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Analyze BitCal-TTS results")
    ap.add_argument("--results-dir", type=str, default="results/raw")
    ap.add_argument("--out-dir", type=str, default="results/processed")
    args = ap.parse_args(argv)

    results_dir = _ROOT / args.results_dir
    out_dir = _ROOT / args.out_dir

    rows = load_all_results(results_dir)
    if not rows:
        print(f"[warn] No result files found in {results_dir}")
        return

    print(f"Loaded {len(rows)} rows from {results_dir}")

    summary = compute_summary(rows)
    print_table(summary)

    csv_path = save_summary_csv(summary, out_dir)
    print(f"Saved summary CSV  -> {csv_path}")

    pareto = plot_pareto(summary, out_dir)
    if pareto:
        print(f"Saved Pareto plot  -> {pareto}")

    bar = plot_bar(summary, out_dir)
    if bar:
        print(f"Saved bar chart    -> {bar}")


if __name__ == "__main__":
    main()
