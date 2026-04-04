"""
Core experiment logic: run one item with a given method, return a result dict.

Three methods are defined:
  "fixed"     -- generate max_new_tokens unconditionally (hard baseline).
  "adaptive"  -- bit-unaware adaptive halting (signals + policy, scale=1.0).
  "bitcal_tts"-- full BitCal-TTS (bit-aware calibration + policy).

Separating logic from the CLI script keeps this testable.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch

from bitcal_tts.calibrator.bit_aware import BitAwareCalibrator
from bitcal_tts.eval.metrics import halting_metrics, summarize_trace
from bitcal_tts.policy.halting import HaltingAction, HaltingPolicy
from bitcal_tts.runner.baseline import ReasoningTrace, run_fixed_budget_loop
from bitcal_tts.runner.budget import TokenBudget
from bitcal_tts.signals.entropy import token_entropy
from bitcal_tts.signals.hidden import hidden_state_stability
from bitcal_tts.signals.trace import reasoning_trace_stability

METHODS = ("fixed", "adaptive", "bitcal_tts")


@dataclass
class ItemResult:
    task_id: str
    method: str
    budget: int
    bit_width: int
    tokens_used: int
    generated_text: str
    predicted_answer: Optional[str]
    gold_answer: Optional[str]
    correct: bool
    actions: List[str] = field(default_factory=list)
    mean_entropy: float = 0.0
    trace_stability: float = 1.0
    hidden_stability: float = 1.0
    n_escalations: int = 0
    n_stops: int = 0
    wall_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Fixed-budget baseline (no step-by-step halting)
# ---------------------------------------------------------------------------

def run_fixed_method(
    model: Any,
    tokenizer: Any,
    task_id: str,
    prompt: str,
    gold_answer: Optional[str],
    budget: int,
    bit_width: int,
    seed: int,
    answer_extractor: Any,
) -> ItemResult:
    """Generate up to `budget` tokens; no halting."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=budget,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    gen_ids = out_ids[0][input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    tokens_used = len(gen_ids)
    pred = answer_extractor(text)
    from benchmarks.gsm8k_loader import answers_match

    correct = answers_match(pred, gold_answer)
    return ItemResult(
        task_id=task_id,
        method="fixed",
        budget=budget,
        bit_width=bit_width,
        tokens_used=tokens_used,
        generated_text=text,
        predicted_answer=pred,
        gold_answer=gold_answer,
        correct=correct,
        wall_ms=wall_ms,
    )


# ---------------------------------------------------------------------------
# Adaptive halting methods
# ---------------------------------------------------------------------------

def run_adaptive_method(
    model: Any,
    tokenizer: Any,
    task_id: str,
    prompt: str,
    gold_answer: Optional[str],
    budget: int,
    method: str,
    bit_width: int,
    policy: HaltingPolicy,
    seed: int,
    step_size: int,
    max_entropy: float,
    answer_extractor: Any,
) -> ItemResult:
    """Run with step-by-step generation + halting policy."""
    from bitcal_tts.integrations.hf_generate import HFStepGenerator

    if method not in ("adaptive", "bitcal_tts"):
        raise ValueError(f"Unsupported method {method!r}")

    eff_bit = bit_width if method == "bitcal_tts" else 16
    calibrator = BitAwareCalibrator(bit_width=eff_bit)

    gen = HFStepGenerator(
        model,
        tokenizer,
        prompt,
        step_size=step_size,
        do_sample=False,
        seed=seed,
        output_hidden_states=True,
    )

    tok_budget = TokenBudget(max_tokens=budget)
    trace = ReasoningTrace()
    actions: List[str] = []
    t0 = time.perf_counter()
    step = 0

    while not tok_budget.exhausted() and not gen.finished:
        text_chunk, logits, hidden, n_tok = gen.step(step, tok_budget)
        if n_tok == 0:
            break
        trace.append_step(text_chunk, logits, hidden, n_tok)
        tok_budget.consume(n_tok)

        ent = token_entropy(logits)
        t_stab = reasoning_trace_stability(trace.texts)
        h_stab = hidden_state_stability(trace.hidden_states)
        conf = calibrator(entropy=ent, trace_stability=t_stab, hidden_stability=h_stab, max_entropy=max_entropy)
        act = policy.decide(entropy=ent, confidence=conf, budget_remaining=tok_budget.remaining)
        actions.append(act.value)

        if act in (HaltingAction.STOP, HaltingAction.ESCALATE):
            break
        step += 1

    wall_ms = (time.perf_counter() - t0) * 1000.0
    full_text = gen.get_full_text()
    pred = answer_extractor(full_text)
    from benchmarks.gsm8k_loader import answers_match

    correct = answers_match(pred, gold_answer)
    summ = summarize_trace(trace)

    return ItemResult(
        task_id=task_id,
        method=method,
        budget=budget,
        bit_width=bit_width,
        tokens_used=trace.total_tokens,
        generated_text=full_text,
        predicted_answer=pred,
        gold_answer=gold_answer,
        correct=correct,
        actions=actions,
        mean_entropy=summ.mean_entropy,
        trace_stability=summ.trace_stability,
        hidden_stability=summ.hidden_stability,
        n_escalations=sum(1 for a in actions if a == "escalate"),
        n_stops=sum(1 for a in actions if a == "stop"),
        wall_ms=wall_ms,
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def run_item(
    model: Any,
    tokenizer: Any,
    task_id: str,
    prompt: str,
    gold_answer: Optional[str],
    *,
    method: str,
    budget: int,
    bit_width: int,
    policy: HaltingPolicy,
    seed: int,
    step_size: int,
    max_entropy: float,
    answer_extractor: Any,
) -> ItemResult:
    if method == "fixed":
        return run_fixed_method(
            model=model,
            tokenizer=tokenizer,
            task_id=task_id,
            prompt=prompt,
            gold_answer=gold_answer,
            budget=budget,
            bit_width=bit_width,
            seed=seed,
            answer_extractor=answer_extractor,
        )
    return run_adaptive_method(
        model=model,
        tokenizer=tokenizer,
        task_id=task_id,
        prompt=prompt,
        gold_answer=gold_answer,
        budget=budget,
        method=method,
        bit_width=bit_width,
        policy=policy,
        seed=seed,
        step_size=step_size,
        max_entropy=max_entropy,
        answer_extractor=answer_extractor,
    )
