# Contributing

Thank you for your interest in BitCal-TTS.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,research]"
python -m pytest tests/ -q
```

The default `pytest` configuration enforces **≥90%** line coverage on `bitcal_tts`. For a faster loop without coverage:

```bash
python -m pytest tests/ -q --no-cov
```

## Pull requests

- Keep changes focused on a single concern.
- Add or update tests for new behavior.
- Run `python -m pytest tests/` before opening a PR.

## Code style

Match existing module layout and typing style. Prefer small, testable functions.

## Research vs. library

Core logic should remain runnable on CPU in CI (mock or tiny models). Heavy model runs and large benchmarks are optional and can live behind scripts or environment flags.
