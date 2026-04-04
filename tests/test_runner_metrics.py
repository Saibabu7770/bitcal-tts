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
