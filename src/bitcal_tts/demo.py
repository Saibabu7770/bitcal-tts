"""Baseline demo: mock reasoning loop + signals + calibration + halting policy."""

from __future__ import annotations

import argparse

from bitcal_tts.calibrator.bit_aware import BitAwareCalibrator
from bitcal_tts.config import load_config
from bitcal_tts.eval.metrics import halting_metrics, summarize_trace
from bitcal_tts.policy.halting import HaltingPolicy
from bitcal_tts.runner.baseline import mock_step_fn, run_fixed_budget_loop
from bitcal_tts.runner.budget import TokenBudget
from bitcal_tts.signals.entropy import token_entropy


def build_demo_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="BitCal-TTS baseline demo (mock steps)")
    ap.add_argument("--config", type=str, default=None, help="YAML config path")
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--bit-width", type=int, default=8)
    return ap


def run_demo(argv: list[str] | None = None) -> None:
    args = build_demo_parser().parse_args(argv)

    policy_kw: dict = {}
    cal_kw: dict = {"bit_width": args.bit_width}
    if args.config:
        cfg = load_config(args.config)
        b = cfg.get("budget", {})
        p = cfg.get("policy", {})
        c = cfg.get("calibrator", {})
        args.budget = int(b.get("max_reasoning_tokens", args.budget))
        policy_kw = {
            "stop_entropy_threshold": float(p.get("stop_entropy_threshold", 2.0)),
            "escalate_entropy_threshold": float(p.get("escalate_entropy_threshold", 4.0)),
            "min_budget_to_continue": int(p.get("min_budget_to_continue", 32)),
        }
        cal_kw["bit_width"] = int(c.get("bit_width_default", args.bit_width))
        cal_kw["temperature"] = float(c.get("temperature", 1.0))

    budget = TokenBudget(max_tokens=args.budget)
    trace = run_fixed_budget_loop(
        budget,
        max_steps=args.max_steps,
        step_fn=mock_step_fn(vocab_size=1024),
    )
    summ = summarize_trace(trace)
    cal = BitAwareCalibrator(**cal_kw)
    policy = HaltingPolicy(**policy_kw)

    actions = []
    br = args.budget
    for _, logits in enumerate(trace.logits_per_step):
        ent = token_entropy(logits)
        conf = cal(
            entropy=ent,
            trace_stability=summ.trace_stability,
            hidden_stability=summ.hidden_stability,
        )
        step_tok = 8
        br -= step_tok
        act = policy.decide(ent, conf, max(0, br))
        actions.append(act.value)

    m = halting_metrics(
        actions=actions,
        final_answer_correct=True,
        tokens_used=trace.total_tokens,
        budget=args.budget,
    )

    print("BitCal-TTS baseline demo (mock)")
    print(f"  steps: {summ.n_steps}, tokens: {summ.total_tokens}")
    print(f"  mean_entropy: {summ.mean_entropy:.4f}")
    print(f"  trace_stability: {summ.trace_stability:.4f}, hidden_stability: {summ.hidden_stability:.4f}")
    print(f"  actions: {actions}")
    print(f"  metrics: {m}")
