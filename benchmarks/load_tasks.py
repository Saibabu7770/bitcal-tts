"""Load benchmark tasks from JSON Lines files (id, prompt, optional answer key)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional


@dataclass
class Task:
    id: str
    prompt: str
    answer: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


def load_jsonl(path: str | Path) -> List[Task]:
    p = Path(path)
    out: List[Task] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(
                Task(
                    id=str(row.get("id", len(out))),
                    prompt=str(row["prompt"]),
                    answer=row.get("answer"),
                    metadata=row.get("metadata"),
                )
            )
    return out


def iter_tasks(path: str | Path) -> Iterator[Task]:
    yield from load_jsonl(path)
