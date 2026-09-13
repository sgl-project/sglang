"""Pinned dataset preparation for speculative-decoding math evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

MATH_INSTRUCTION = "Please reason step by step and put your final answer in \\boxed{}."


class BenchmarkConfig(NamedTuple):
    name: str
    expected_rows: int
    instruction: str
    chat_template_kwargs: dict[str, object]


BENCHMARKS = {
    name: BenchmarkConfig(
        name=name,
        expected_rows=expected_rows,
        instruction=MATH_INSTRUCTION,
        chat_template_kwargs={"reasoning_effort": "high"},
    )
    for name, expected_rows in (("math500", 500),)
}

DATASET_REVISIONS = {
    "HuggingFaceH4/MATH-500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
}


def get_benchmark(name: str) -> BenchmarkConfig:
    try:
        return BENCHMARKS[name]
    except KeyError as exc:
        available = ", ".join(BENCHMARKS)
        raise KeyError(f"Unknown benchmark {name!r}. Available: {available}") from exc


def _load_dataset(
    repo_id: str,
    config_name: str | None = None,
    *,
    split: str,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        repo_id,
        config_name,
        split=split,
        revision=DATASET_REVISIONS[repo_id],
    )
    return [dict(row) for row in dataset]


def _prepare_math500() -> list[dict[str, Any]]:
    rows = _load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [
        {
            "row": index,
            "ground_truth": row["answer"],
            "chat_input": [{"role": "user", "content": row["problem"]}],
        }
        for index, row in enumerate(rows)
    ]


BUILDERS = {
    "math500": _prepare_math500,
}


def _row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def prepare_benchmark_data(name: str, *, output_dir: Path) -> Path:
    benchmark = get_benchmark(name)
    output = output_dir / f"{name}.jsonl"
    if output.is_file():
        count = _row_count(output)
        if count == benchmark.expected_rows:
            return output
        raise ValueError(
            f"{output} has {count} rows; expected {benchmark.expected_rows}"
        )

    records = BUILDERS[name]()
    if len(records) != benchmark.expected_rows:
        raise ValueError(
            f"Expected {benchmark.expected_rows} rows for {name}, got {len(records)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return output
