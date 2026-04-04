"""Command-line interface for BitCal-TTS."""

from __future__ import annotations

import argparse
import sys

from bitcal_tts import __version__
from bitcal_tts.demo import run_demo


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="bitcal-tts",
        description="Bit-calibrated test-time scaling for quantized reasoning models.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", help="Subcommands")

    p_demo = sub.add_parser("demo", help="Run mock baseline demo (no GPU model required)")
    p_demo.add_argument("--config", type=str, default=None)
    p_demo.add_argument("--max-steps", type=int, default=5)
    p_demo.add_argument("--budget", type=int, default=256)
    p_demo.add_argument("--bit-width", type=int, default=8)

    p_doc = sub.add_parser("doctor", help="Print dependency availability")

    p_hf = sub.add_parser(
        "hf-smoke",
        help="Single forward pass with a HF causal LM (downloads weights; needs transformers)",
    )
    p_hf.add_argument("--model", type=str, default="gpt2", help="HF model id or path")
    p_hf.add_argument("--prompt", type=str, default="Hello")

    args = parser.parse_args(argv)

    if args.command is None or args.command == "demo":
        if args.command is None:
            run_demo([])
            return
        run_demo(
            [
                *(["--config", args.config] if args.config else []),
                "--max-steps",
                str(args.max_steps),
                "--budget",
                str(args.budget),
                "--bit-width",
                str(args.bit_width),
            ]
        )
        return

    if args.command == "doctor":
        _print_doctor()
        return

    if args.command == "hf-smoke":
        from bitcal_tts.integrations.hf_inference import hf_smoke_forward

        hf_smoke_forward(args.model, args.prompt)
        return

    parser.print_help()


def _print_doctor() -> None:
    lines = [f"BitCal-TTS {__version__}", ""]
    try:
        import torch

        lines.append(f"  torch: {torch.__version__} (cuda available: {torch.cuda.is_available()})")
    except ImportError:
        lines.append("  torch: not installed")

    try:
        import transformers

        lines.append(f"  transformers: {transformers.__version__}")
    except ImportError:
        lines.append("  transformers: not installed (optional; pip install bitcal-tts[research])")

    try:
        import yaml

        lines.append(f"  pyyaml: {yaml.__version__}")
    except ImportError:
        lines.append("  pyyaml: not installed")

    print("\n".join(lines))
