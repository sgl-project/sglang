"""Aggregate per-model SSM-dtype runs into a single matrix summary.

Reads ``runs.tsv`` written by the matrix shell script, then loads each run's
``summary.json``; writes ``matrix_summary.{json,csv,md}`` to the matrix output
root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoring  # noqa: E402


def load_rows(runs_tsv: Path) -> list[dict]:
    rows: list[dict] = []
    with runs_tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for run in reader:
            summary_path = Path(run["output_dir"]) / "summary.json"
            if not summary_path.exists():
                rows.append(
                    {
                        **run,
                        "dtype": None,
                        "eval": None,
                        "score": None,
                        "status": "missing_summary",
                    }
                )
                continue
            summary = json.loads(summary_path.read_text())
            for row in summary.get("rows", []):
                rows.append({**run, **row, "status": "ok"})
    return rows


def build_score_index(rows: list[dict]):
    scores: dict[tuple, float | None] = {}
    dtypes_by_model: dict[str, set[str]] = {}
    evals_by_model: dict[str, set[str]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        model = row["model_label"]
        dtype = row["dtype"]
        eval_name = row["eval"]
        scores[(model, dtype, eval_name)] = row["score"]
        dtypes_by_model.setdefault(model, set()).add(dtype)
        evals_by_model.setdefault(model, set()).add(eval_name)
    return scores, dtypes_by_model, evals_by_model


def render_markdown(rows, scores, dtypes_by_model, evals_by_model) -> str:
    lines = ["# Linear Attention SSM Dtype Accuracy Matrix", ""]
    for model in sorted(dtypes_by_model):
        dtypes = scoring.sort_dtypes_float32_first(dtypes_by_model[model])
        evals = sorted(evals_by_model[model])
        lines += [f"## {model}", ""]
        # Matrix score table uses right-aligned columns; render manually to
        # preserve that distinction (the per-model table uses left-aligned).
        lines.append("| Eval | " + " | ".join(dtypes) + " |")
        lines.append("|---" + "|---:" * len(dtypes) + "|")
        for eval_name in evals:
            vals = [
                scoring.format_score(scores.get((model, dtype, eval_name)))
                for dtype in dtypes
            ]
            lines.append(f"| {eval_name} | " + " | ".join(vals) + " |")
        if "float32" in dtypes:
            lines.append("")
            lines.extend(
                scoring.render_delta_table_matrix(scores, model, dtypes, evals)
            )
        lines.append("")

    lines += ["## Runs", ""]
    for row in rows:
        if row.get("status") == "ok":
            continue
        lines.append(
            f"- {row['model_label']}: {row['status']} at `{row['output_dir']}`"
        )
    return "\n".join(lines) + "\n"


def write_outputs(matrix_output_root: Path, rows: list[dict]) -> None:
    (matrix_output_root / "matrix_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    with (matrix_output_root / "matrix_summary.csv").open("w", newline="") as f:
        fieldnames = [
            "model_key",
            "model_label",
            "model_path",
            "dtype",
            "eval",
            "score",
            "latency",
            "output_throughput",
            "metrics_file",
            "output_dir",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    scores, dtypes_by_model, evals_by_model = build_score_index(rows)
    md = render_markdown(rows, scores, dtypes_by_model, evals_by_model)
    (matrix_output_root / "matrix_summary.md").write_text(md)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-tsv", default=os.environ.get("RUNS_TSV"))
    parser.add_argument(
        "--matrix-output-root", default=os.environ.get("MATRIX_OUTPUT_ROOT")
    )
    args = parser.parse_args(argv)
    if not args.runs_tsv or not args.matrix_output_root:
        parser.error("--runs-tsv and --matrix-output-root are required")
    runs_tsv = Path(args.runs_tsv)
    matrix_output_root = Path(args.matrix_output_root)
    rows = load_rows(runs_tsv)
    write_outputs(matrix_output_root, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
