from pathlib import Path

from benchmarks.load_tasks import load_jsonl


def test_load_example_jsonl():
    root = Path(__file__).resolve().parents[1]
    tasks = load_jsonl(root / "benchmarks" / "example_tasks.jsonl")
    assert len(tasks) == 2
    assert tasks[0].id == "ex-1"
    assert "train" in tasks[0].prompt
