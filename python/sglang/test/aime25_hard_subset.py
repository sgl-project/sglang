"""Utilities for building a small hard AIME25 subset for CI.

The subset is meant for fast DeepSeek-V4 accuracy probes where the full AIME25
run is too expensive. Rows use the NeMo-Skills/sgl-eval JSONL shape:
``{"id": ..., "problem": ..., "expected_answer": ...}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_DSV4_PRO_HARD_IDS: tuple[str, ...] = (
    "aime25-13",
    "aime25-14",
    "aime25-27",
    "aime25-29",
)

DEFAULT_DSV4_FLASH_HARD_IDS = DEFAULT_DSV4_PRO_HARD_IDS

DSV4_FLASH_BISECT_IDS: tuple[str, ...] = (
    "aime25-13",
    "aime25-14",
    "aime25-29",
)

PRESET_IDS: dict[str, tuple[str, ...]] = {
    "dsv4-flash": DEFAULT_DSV4_FLASH_HARD_IDS,
    "dsv4-flash-bisect": DSV4_FLASH_BISECT_IDS,
    "dsv4-pro": DEFAULT_DSV4_PRO_HARD_IDS,
}


def parse_problem_ids(problem_ids: str | Iterable[str] | None) -> tuple[str, ...]:
    if problem_ids is None:
        return DEFAULT_DSV4_FLASH_HARD_IDS
    if isinstance(problem_ids, str):
        values = problem_ids.split(",")
    else:
        values = list(problem_ids)
    parsed = tuple(x.strip() for x in values if x and x.strip())
    if not parsed:
        raise ValueError("at least one AIME25 problem id is required")
    return parsed


def normalize_aime25_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        problem = row.get("problem") or row.get("question")
        expected_answer = row.get("expected_answer")
        if expected_answer is None:
            expected_answer = row.get("answer")
        if problem is None or expected_answer is None:
            raise ValueError(
                f"row {idx} must include problem/question and expected_answer/answer"
            )
        normalized.append(
            {
                "id": str(row.get("id") or f"aime25-{idx}"),
                "problem": str(problem),
                "expected_answer": str(expected_answer),
            }
        )
    return normalized


def select_aime25_rows(
    rows: Iterable[Mapping[str, object]],
    problem_ids: Sequence[str],
) -> list[dict[str, str]]:
    normalized = normalize_aime25_rows(rows)
    by_id = {row["id"]: row for row in normalized}
    missing = [problem_id for problem_id in problem_ids if problem_id not in by_id]
    if missing:
        available = ", ".join(sorted(by_id))
        raise ValueError(
            f"AIME25 problem id(s) not found: {', '.join(missing)}. "
            f"Available ids: {available}"
        )
    return [by_id[problem_id] for problem_id in problem_ids]


def write_jsonl(rows: Iterable[Mapping[str, object]], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return out_path


def load_aime25_rows_from_huggingface() -> list[dict[str, str]]:
    from datasets import load_dataset

    rows = []
    for config in ("AIME2025-I", "AIME2025-II"):
        rows.extend(load_dataset("opencompass/AIME2025", config, split="test"))
    return normalize_aime25_rows(
        {
            "id": f"aime25-{idx}",
            "problem": row["question"],
            "expected_answer": str(row["answer"]),
        }
        for idx, row in enumerate(rows)
    )


def build_aime25_hard_subset(
    path: str | Path,
    *,
    rows: Iterable[Mapping[str, object]] | None = None,
    problem_ids: Sequence[str] | str | None = None,
) -> Path:
    source_rows = rows if rows is not None else load_aime25_rows_from_huggingface()
    selected = select_aime25_rows(source_rows, parse_problem_ids(problem_ids))
    return write_jsonl(selected, path)
