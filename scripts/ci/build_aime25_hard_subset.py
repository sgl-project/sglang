#!/usr/bin/env python3
"""Build a hard AIME25 JSONL subset for sgl-eval CI probes.

Example:
  python3 scripts/ci/build_aime25_hard_subset.py \
    --out /tmp/aime25-hard.jsonl

Then run:
  sgl-eval run aime25 --from-dataset /tmp/aime25-hard.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.test.aime25_hard_subset import (
    PRESET_IDS,
    parse_problem_ids,
    select_aime25_rows,
    write_jsonl,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _load_from_sgl_eval() -> list[dict]:
    from sgl_eval.evals._loader import load_bundled

    examples = load_bundled("aime25")(None)
    return [
        {
            "id": ex.id,
            "problem": ex.inputs["problem"],
            "expected_answer": ex.target,
        }
        for ex in examples
    ]


def _load_from_huggingface() -> list[dict]:
    from datasets import load_dataset

    rows = []
    for config in ("AIME2025-I", "AIME2025-II"):
        rows.extend(load_dataset("opencompass/AIME2025", config, split="test"))
    return [
        {
            "id": f"aime25-{idx}",
            "problem": row["question"],
            "expected_answer": str(row["answer"]),
        }
        for idx, row in enumerate(rows)
    ]


def load_aime25_rows(source_jsonl: str | None) -> list[dict]:
    if source_jsonl:
        return _read_jsonl(Path(source_jsonl).expanduser())

    errors = []
    for loader in (_load_from_sgl_eval, _load_from_huggingface):
        try:
            return loader()
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")
    raise SystemExit(
        "failed to load AIME25 rows; install sgl-eval or datasets, or pass "
        f"--source-jsonl. Errors: {'; '.join(errors)}"
    )


def resolve_problem_ids(args: argparse.Namespace) -> tuple[str, ...]:
    if args.problem_ids:
        return parse_problem_ids(args.problem_ids)
    try:
        return PRESET_IDS[args.preset]
    except KeyError as exc:
        choices = ", ".join(sorted(PRESET_IDS))
        raise SystemExit(f"unknown preset {args.preset!r}; choices: {choices}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path for sgl-eval --from-dataset.",
    )
    parser.add_argument(
        "--preset",
        default="dsv4-flash",
        help=f"Hard subset preset. Choices: {', '.join(sorted(PRESET_IDS))}.",
    )
    parser.add_argument(
        "--problem-ids",
        help="Comma-separated AIME25 ids, overriding --preset.",
    )
    parser.add_argument(
        "--source-jsonl",
        help=(
            "Optional source JSONL with problem/question and expected_answer/answer "
            "fields. By default the script loads the bundled sgl-eval AIME25 data, "
            "falling back to opencompass/AIME2025."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    problem_ids = resolve_problem_ids(args)
    rows = select_aime25_rows(load_aime25_rows(args.source_jsonl), problem_ids)
    out_path = write_jsonl(rows, args.out)
    print(
        f"Wrote {len(rows)} AIME25 hard problem(s) to {out_path}: "
        f"{', '.join(row['id'] for row in rows)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
