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
    for name, expected_rows in (
        ("gsm8k", 1319),
        ("math500", 500),
        ("aime24", 30),
        ("aime25", 30),
        ("aime26", 30),
    )
}

DATASET_REVISIONS = {
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "HuggingFaceH4/MATH-500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "hypaai/Hypa_AIME2024": "11ab79f0eed5f4fdf3d469b466663ab86bbd77c8",
    "math-ai/aime25": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
    "math-ai/aime26": "79037aebdb6580008fb960d17cb21fd3099083e3",
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


def _prepare_gsm8k() -> list[dict[str, Any]]:
    rows = _load_dataset("openai/gsm8k", "main", split="test")
    records = []
    for index, row in enumerate(rows):
        prompt = f"Q: {row['question']}\nA: Let's think step by step."
        records.append(
            {
                "row": index,
                "ground_truth": row["answer"],
                "chat_input": [{"role": "user", "content": prompt}],
            }
        )
    return records


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


def _prepare_aime(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(rows):
        prompt = f"Question: {row['problem']}\nAnswer:"
        records.append(
            {
                "row": index,
                "ground_truth": str(row.get("answer", row.get("solution"))).strip(),
                "chat_input": [{"role": "user", "content": prompt}],
            }
        )
    return records


def _prepare_aime24() -> list[dict[str, Any]]:
    return _prepare_aime(_load_dataset("hypaai/Hypa_AIME2024", split="english"))


def _prepare_aime25() -> list[dict[str, Any]]:
    return _prepare_aime(_load_dataset("math-ai/aime25", split="test"))


def _prepare_aime26() -> list[dict[str, Any]]:
    return _prepare_aime(_load_dataset("math-ai/aime26", split="test"))


BUILDERS = {
    "gsm8k": _prepare_gsm8k,
    "math500": _prepare_math500,
    "aime24": _prepare_aime24,
    "aime25": _prepare_aime25,
    "aime26": _prepare_aime26,
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
