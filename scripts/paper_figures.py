#!/usr/bin/env python3
"""
Publication-style figures for the BitCal-TTS paper (~10-page arXiv draft).

Writes vector PDFs (and PNG) under repo `media/` for inclusion via \\graphicspath{{media/}}.

Numbers for cross-model panels match the paper's main GSM8K table (budget 512, 4-bit, n=100).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MEDIA = _ROOT / "media"


def _style_mpl() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman", "Times", "serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
        }
    )


def fig_main_accuracy_tokens() -> None:
    """Grouped bars: accuracy and avg tokens for 3B/7B/14B at budget 512."""
    import matplotlib.pyplot as plt
    import numpy as np

    models = ["3B", "7B", "14B"]
    methods = ["fixed", "adaptive", "BitCal-TTS"]
    acc = {
        "fixed": [60.0, 90.7, 88.6],
        "adaptive": [22.0, 79.6, 82.9],
        "BitCal-TTS": [20.0, 83.3, 85.7],
    }
    toks = {
        "fixed": [281, 466, 455],
        "adaptive": [132, 286, 239],
        "BitCal-TTS": [144, 316, 269],
    }
    colors = {"fixed": "#4C72B0", "adaptive": "#DD8452", "BitCal-TTS": "#55A868"}
    x = np.arange(len(models))
    width = 0.24

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 4.25), constrained_layout=True)
    for i, m in enumerate(methods):
        off = (i - 1) * width
        ax0.bar(x + off, acc[m], width, label=m, color=colors[m], edgecolor="black", linewidth=0.35)
        ax1.bar(x + off, toks[m], width, label=m, color=colors[m], edgecolor="black", linewidth=0.35)

    for ax in (ax0, ax1):
        ax.set_xticks(x)
        ax.set_xticklabels([f"Qwen2.5-{s}-Instruct" for s in models], rotation=12, ha="right")

    ax0.set_ylabel("GSM8K accuracy (%)")
    ax0.set_title("Accuracy at token budget 512 (4-bit)")
    ax0.set_ylim(0, 100)
    ax0.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)

    ax1.set_ylabel("Average tokens used")
    ax1.set_title("Compute use at token budget 512 (4-bit)")
    ax1.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)

    out_pdf = _MEDIA / "fig_main_accuracy_tokens.pdf"
    out_png = _MEDIA / "fig_main_accuracy_tokens.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def fig_pareto_7b_from_csv() -> None:
    """Pareto-style scatter for Qwen2.5-7B if processed summary exists; else skip."""
    import csv

    csv_path = _ROOT / "results" / "processed" / "7b" / "summary.csv"
    if not csv_path.is_file():
        print(f"[skip] No {csv_path} for Pareto figure")
        return

    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    import matplotlib.pyplot as plt

    palette = {"fixed": "#4C72B0", "adaptive": "#DD8452", "bitcal_tts": "#55A868"}
    marker = {"fixed": "o", "adaptive": "s", "bitcal_tts": "^"}

    fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)
    seen: set[str] = set()
    for row in rows:
        m = row["method"]
        lbl = m if m not in seen else None
        seen.add(m)
        ax.scatter(
            float(row["avg_tokens"]),
            float(row["accuracy"]),
            c=palette.get(m, "#7f7f7f"),
            marker=marker.get(m, "o"),
            s=70,
            edgecolors="black",
            linewidths=0.35,
            alpha=0.92,
            label=lbl,
        )
        ax.annotate(
            row["budget"],
            (float(row["avg_tokens"]), float(row["accuracy"])),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=8,
        )

    ax.set_xlabel("Average tokens used")
    ax.set_ylabel("Accuracy")
    ax.set_title("Quality–efficiency trade-off (Qwen2.5-7B, GSM8K)")
    ax.legend(loc="lower right", frameon=False)

    out_pdf = _MEDIA / "fig_pareto_7b.pdf"
    out_png = _MEDIA / "fig_pareto_7b.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def fig_7b_budget_sweep_from_csv() -> None:
    """Two-panel line plot: accuracy and avg tokens vs budget for Qwen2.5-7B (processed CSV)."""
    import csv

    csv_path = _ROOT / "results" / "processed" / "7b" / "summary.csv"
    if not csv_path.is_file():
        print(f"[skip] No {csv_path} for 7B budget sweep figure")
        return

    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    import matplotlib.pyplot as plt
    import numpy as np

    palette = {"fixed": "#4C72B0", "adaptive": "#DD8452", "bitcal_tts": "#55A868"}
    label = {"fixed": "Fixed", "adaptive": "Adaptive", "bitcal_tts": "BitCal-TTS"}
    by_method: dict[str, dict[int, tuple[float, float]]] = {}
    for row in rows:
        m = row["method"]
        b = int(row["budget"])
        acc = float(row["accuracy"]) * 100.0
        tok = float(row["avg_tokens"])
        by_method.setdefault(m, {})[b] = (acc, tok)

    budgets = sorted({int(r["budget"]) for r in rows})
    xs = np.array(budgets, dtype=float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 4.25), constrained_layout=True)
    for m in ("fixed", "adaptive", "bitcal_tts"):
        if m not in by_method:
            continue
        accs = [by_method[m][b][0] for b in budgets]
        toks = [by_method[m][b][1] for b in budgets]
        kw = dict(
            marker="o",
            ms=7,
            lw=2.0,
            color=palette[m],
            label=label[m],
            markeredgecolor="black",
            markeredgewidth=0.35,
        )
        ax0.plot(xs, accs, **kw)
        ax1.plot(xs, toks, **kw)

    for ax in (ax0, ax1):
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(x)) for x in xs])

    ax0.set_xlabel("Token budget $B$ (max new tokens)")
    ax0.set_ylabel("GSM8K accuracy (%)")
    ax0.set_title("Qwen2.5-7B-Instruct (4-bit): accuracy vs budget")
    ax0.set_ylim(35, 98)
    ax0.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.4")

    ax1.set_xlabel("Token budget $B$ (max new tokens)")
    ax1.set_ylabel("Average tokens used")
    ax1.set_title("Qwen2.5-7B-Instruct (4-bit): compute vs budget")
    ax1.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="0.4")

    out_pdf = _MEDIA / "fig_7b_budget_sweep.pdf"
    out_png = _MEDIA / "fig_7b_budget_sweep.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def fig_premature_stop() -> None:
    """Premature-stop rate at budget 512 (paper table)."""
    import matplotlib.pyplot as plt
    import numpy as np

    models = ["3B", "7B", "14B"]
    adaptive = [63.0, 14.8, 17.1]
    bitcal = [63.0, 11.1, 11.4]
    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.2, 3.85), constrained_layout=True)
    ax.bar(x - w / 2, adaptive, w, label="adaptive", color="#DD8452", edgecolor="black", linewidth=0.35)
    ax.bar(x + w / 2, bitcal, w, label="BitCal-TTS", color="#55A868", edgecolor="black", linewidth=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Qwen2.5-{s}" for s in models])
    ax.set_ylabel("Premature-stop rate (%)")
    ax.set_title("Early halt and wrong (budget 512)")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(adaptive + bitcal) * 1.12)

    out_pdf = _MEDIA / "fig_premature_stop.pdf"
    out_png = _MEDIA / "fig_premature_stop.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def fig_full_system_architecture() -> None:
    """
    Minimal diagram: wide spacing, short labels, numbered flow key (no text on arrows).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    pub_rc = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    }

    C_IO = "#ececec"
    C_LM = "#d5e5f7"
    C_CTRL_BG = "#fbf8f4"
    C_SIGNALS = "#ffcc88"
    C_CALIB = "#ffaa55"
    C_POLICY = "#b5e0b8"
    C_CONFIRM = "#c5daf2"
    C_EDGE = "#2a2a2a"
    C_ARROW = "#333333"
    C_STREAM = "#0d4a6e"

    def box(ax, xy, w, h, body, *, title, fc, fs=9.8, tfs=10.8):
        x, y = xy
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.025,rounding_size=0.22",
                linewidth=1.0,
                edgecolor=C_EDGE,
                facecolor=fc,
                zorder=2,
            )
        )
        cx = x + w / 2
        ax.text(cx, y + h - 0.32, title, ha="center", va="top", fontsize=tfs, fontweight="bold", color=C_EDGE, zorder=3)
        ax.text(cx, y + h * 0.42, body, ha="center", va="center", fontsize=fs, linespacing=1.35, zorder=3)

    def arr(ax, p0, p1, *, lw=1.35, ms=12, c=C_ARROW, z=4):
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=ms,
                linewidth=lw,
                color=c,
                shrinkA=8,
                shrinkB=8,
                zorder=z,
            )
        )

    with plt.rc_context(pub_rc):
        fig, ax = plt.subplots(figsize=(15.5, 9.0), dpi=150)
        ax.set_xlim(0, 36.0)
        ax.set_ylim(-0.35, 15.0)
        ax.axis("off")
        fig.patch.set_facecolor("white")

        # --- Top row: main data path (single y for arrows) ---
        y_row = 10.35
        h_io = 2.35
        h_lm = 2.65
        x_in, w_in = 1.0, 5.2
        x_lm, w_lm = 7.0, 8.8
        x_out, w_out = 27.0, 5.8
        y_in = y_row + (h_lm - h_io) / 2
        y_out = y_in

        box(ax, (x_in, y_in), w_in, h_io, r"Formatted prompt $x$", title="Input", fc=C_IO, fs=10.2)
        box(
            ax,
            (x_lm, y_row),
            w_lm,
            h_lm,
            r"4-bit causal LM" + "\n" + r"$k$ tokens / step (greedy)",
            title="Quantized LM",
            fc=C_LM,
            fs=10.0,
        )
        box(ax, (x_out, y_out), w_out, h_io, r"Trace + parsed answer", title="Output", fc=C_IO, fs=10.2)

        y_a = y_row + h_lm * 0.55
        arr(ax, (x_in + w_in, y_a), (x_lm - 0.08, y_a), lw=1.5, c=C_STREAM)
        arr(ax, (x_lm + w_lm, y_a), (x_out - 0.08, y_a), lw=2.0, ms=14, c=C_STREAM)

        # Tiny on-arrow markers only (numbers), no long captions
        for (px, py), t in (
            (((x_in + w_in + x_lm) / 2, y_a + 0.55), "1"),
            (((x_lm + w_lm + x_out) / 2, y_a + 0.55), "2"),
        ):
            ax.text(
                px,
                py,
                t,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=C_STREAM,
                zorder=8,
            )

        # --- Controller (below, large margin from LM) ---
        ctrl_x, ctrl_y = 2.2, 1.05
        ctrl_w, ctrl_h = 31.6, 7.35
        ctrl_top = ctrl_y + ctrl_h
        ax.add_patch(
            FancyBboxPatch(
                (ctrl_x, ctrl_y),
                ctrl_w,
                ctrl_h,
                boxstyle="round,pad=0.035,rounding_size=0.32",
                linewidth=1.2,
                edgecolor=C_EDGE,
                facecolor=C_CTRL_BG,
                zorder=1,
            )
        )
        ax.text(
            ctrl_x + ctrl_w / 2,
            ctrl_top + 0.28,
            "BitCal-TTS controller",
            ha="center",
            va="bottom",
            fontsize=12.5,
            fontweight="bold",
            color=C_EDGE,
            zorder=3,
        )

        pad = 0.75
        gap_m = 0.55
        w_m = (ctrl_w - 2 * pad - 2 * gap_m) / 3
        xi = ctrl_x + pad
        ym, hm = ctrl_y + 3.55, 3.25

        box(
            ax,
            (xi, ym),
            w_m,
            hm,
            r"$H_t$, $\tau^{\mathrm{tr}}_t$, $\tau^{\mathrm{hid}}_t$",
            title="Signals",
            fc=C_SIGNALS,
            fs=10.5,
        )
        box(
            ax,
            (xi + w_m + gap_m, ym),
            w_m,
            hm,
            r"$c_t = \mathrm{clip}(c^{\mathrm{raw}} s(b), 0, 1)$",
            title="Calibrator",
            fc=C_CALIB,
            fs=10.0,
        )
        box(
            ax,
            (xi + 2 * (w_m + gap_m), ym),
            w_m,
            hm,
            r"$\theta_H$, $\theta_c$, $m$, $B$",
            title="Halting",
            fc=C_POLICY,
            fs=10.5,
        )

        yc, hc = ctrl_y + 0.95, 2.45
        box(
            ax,
            (xi, yc),
            ctrl_w - 2 * pad,
            hc,
            r"After ####: wait $\Delta(b)$ tokens" + "\n" + r"$\Delta(4)=32$; adaptive $b_{\mathrm{eff}}=16 \Rightarrow \Delta=0$",
            title="Answer-marker tail",
            fc=C_CONFIRM,
            fs=9.9,
        )

        # LM ↔ controller: wide vertical channel, no side text
        x_dn = x_lm + w_lm * 0.28
        x_up = x_lm + w_lm * 0.72
        arr(ax, (x_dn, y_row), (x_dn, ctrl_top), lw=1.4, ms=12)
        arr(ax, (x_up, ctrl_top), (x_up, y_row), lw=1.4, ms=12)
        y_mid = (y_row + ctrl_top) / 2
        ax.text(x_dn - 0.55, y_mid, "3", ha="center", va="center", fontsize=11, fontweight="bold", color=C_ARROW, zorder=8)
        ax.text(x_up + 0.55, y_mid, "4", ha="center", va="center", fontsize=11, fontweight="bold", color=C_ARROW, zorder=8)

        # Flow key (single block, below diagram)
        key_y = 0.55
        key = (
            r"$\mathbf{1}$ prompt \quad $\mathbf{2}$ decoded tokens \quad "
            r"$\mathbf{3}$ logits / last hidden \quad $\mathbf{4}$ continue $\mid$ stop $\mid$ escalate"
        )
        ax.text(
            18.0,
            key_y + 0.22,
            key,
            ha="center",
            va="center",
            fontsize=10.0,
            color=C_EDGE,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#bbbbbb", linewidth=0.8),
            zorder=5,
        )
        ax.text(
            18.0,
            key_y - 0.12,
            "Frozen weights. Sidecar on Hugging Face 4-bit inference.",
            ha="center",
            va="top",
            fontsize=9.0,
            style="italic",
            color="#555555",
            zorder=5,
        )

        ax.text(
            18.0,
            14.35,
            "BitCal-TTS: system architecture",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=C_EDGE,
            zorder=30,
        )

        fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03)
        base = _MEDIA / "architecture_bitcal_tts_full"
        fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=600, facecolor="white", edgecolor="none")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)



def main() -> None:
    _MEDIA.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    _style_mpl()
    fig_main_accuracy_tokens()
    fig_premature_stop()
    fig_pareto_7b_from_csv()
    fig_7b_budget_sweep_from_csv()
    fig_full_system_architecture()
    print(f"Wrote figures under {_MEDIA}")


if __name__ == "__main__":
    main()
