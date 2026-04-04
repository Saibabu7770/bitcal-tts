import torch

from bitcal_tts.eval.metrics import halting_metrics, summarize_trace
from bitcal_tts.runner.baseline import mock_step_fn, run_fixed_budget_loop
from bitcal_tts.runner.budget import TokenBudget


def test_fixed_budget_loop_runs():
    b = TokenBudget(max_tokens=100)
    trace = run_fixed_budget_loop(b, max_steps=3, step_fn=mock_step_fn(vocab_size=128))
    assert len(trace.texts) == 3
    assert trace.total_tokens == 24
    summ = summarize_trace(trace)
    assert summ.n_steps == 3
    assert summ.mean_entropy >= 0.0


def test_halting_metrics_keys():
    m = halting_metrics(
        actions=["continue", "stop"],
        final_answer_correct=False,
        tokens_used=50,
        budget=100,
    )
    assert m["accuracy"] == 0.0
    assert m["tokens_used"] == 50
    assert m["n_stops"] == 1


def test_loop_zero_budget_no_steps():
    b = TokenBudget(max_tokens=0)
    trace = run_fixed_budget_loop(b, max_steps=5, step_fn=mock_step_fn(vocab_size=64))
    assert len(trace.texts) == 0


def test_loop_partial_steps_until_budget():
    b = TokenBudget(max_tokens=10)

    def step_fn(step, _budget):
        return "x", torch.zeros(8), None, 3

    trace = run_fixed_budget_loop(b, max_steps=100, step_fn=step_fn)
    # 3+3+3 = 9 ok, next would be 12 > 10 -> after 3 steps used=9, remaining=1; step 4: consume 3 -> 12 used but we check exhausted at start of iteration
    # Actually after step 3: used=9, remaining=1. Step 4: not exhausted, consume 3 -> used=12, remaining max(0,-2)=0
    assert len(trace.texts) == 4
