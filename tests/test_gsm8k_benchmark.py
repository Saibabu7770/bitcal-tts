"""Tests for benchmarks/gsm8k_loader.py."""

import json
from pathlib import Path

import pytest

from benchmarks.gsm8k_loader import (
    GSM8KItem,
    GSM8K_PROMPT_TEMPLATE,
    answers_match,
    extract_answer,
    load_gsm8k_jsonl,
)


class TestExtractAnswer:
    def test_hash_format(self):
        assert extract_answer("The answer is #### 80") == "80"

    def test_hash_format_with_comma(self):
        assert extract_answer("#### 1,200") == "1200"

    def test_negative_answer(self):
        assert extract_answer("#### -5") == "-5"

    def test_fallback_last_number(self):
        assert extract_answer("So we get 42 cookies.") == "42"

    def test_no_number_returns_none(self):
        assert extract_answer("No numbers here.") is None

    def test_float_answer(self):
        result = extract_answer("#### 3.5")
        assert result == "3.5"


class TestAnswersMatch:
    def test_exact_match(self):
        assert answers_match("80", "80") is True

    def test_numeric_equal(self):
        assert answers_match("1200", "1200") is True

    def test_numeric_mismatch(self):
        assert answers_match("80", "81") is False

    def test_none_returns_false(self):
        assert answers_match(None, "80") is False
        assert answers_match("80", None) is False

    def test_float_tolerance(self):
        assert answers_match("1.0000001", "1.0") is True


class TestLoadJsonl:
    def test_load_example_tasks(self):
        root = Path(__file__).resolve().parents[1]
        tasks = load_gsm8k_jsonl(root / "benchmarks" / "example_tasks.jsonl")
        assert len(tasks) >= 1
        assert isinstance(tasks[0], GSM8KItem)
        assert tasks[0].prompt != ""

    def test_load_custom_jsonl(self, tmp_path):
        p = tmp_path / "tasks.jsonl"
        rows = [
            {"id": "t1", "question": "What is 2+2?", "answer": "#### 4"},
            {"id": "t2", "question": "What is 3*3?", "answer": "#### 9"},
        ]
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        items = load_gsm8k_jsonl(p)
        assert len(items) == 2
        assert items[0].gold_answer == "4"
        assert items[1].gold_answer == "9"

    def test_n_samples_limit(self, tmp_path):
        p = tmp_path / "tasks.jsonl"
        with p.open("w") as f:
            for i in range(10):
                f.write(json.dumps({"id": str(i), "question": f"Q{i}", "answer": f"#### {i}"}) + "\n")
        items = load_gsm8k_jsonl(p, n_samples=3)
        assert len(items) == 3

    def test_prompt_template_applied(self, tmp_path):
        p = tmp_path / "tasks.jsonl"
        with p.open("w") as f:
            f.write(json.dumps({"id": "1", "question": "What is 1+1?"}) + "\n")
        items = load_gsm8k_jsonl(p)
        assert "1+1" in items[0].prompt
        assert "####" in items[0].prompt
