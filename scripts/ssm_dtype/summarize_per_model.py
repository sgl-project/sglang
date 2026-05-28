"""Aggregate per-dtype eval metrics for a single model run.

Reads ``${OUTPUT_DIR}/${dtype}_${eval}.metrics.json`` produced by
``sglang.test.run_eval``; writes ``summary.json``, ``summary.csv`` and
``summary.md`` in the same directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Allow ``python scripts/ssm_dtype/summarize_per_model.py`` without setting
# PYTHONPATH by adding our own directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoring  # noqa: E402


def collect_rows(output_dir: Path, dtypes: list[str], evals: list[str]):
    rows = []
    scores: dict[tuple[str, str], float | None] = {}
    for dtype in dtypes:
        for eval_name in evals:
            path = output_dir / f"{dtype}_{eval_name}.metrics.json"
            if not path.exists():
                continue
            metrics = json.loads(path.read_text())
            score = scoring.extract_score(metrics)
            rows.append(
                {
                    "dtype": dtype,
                    "eval": eval_name,
                    "score": score,
                    "latency": metrics.get("latency"),
                    "output_throughput": metrics.get("output_throughput"),
                    "metrics_file": path.name,
                }
            )
            scores[(dtype, eval_name)] = score
    return rows, scores


def write_outputs(
    output_dir: Path,
    model_label: str,
    dtypes: list[str],
    evals: list[str],
    rows: list[dict],
    scores: dict,
) -> None:
    baseline_dtype = scoring.select_baseline_dtype(dtypes)
    deltas = scoring.compute_deltas(scores, dtypes, evals, baseline_dtype)
    summary = {"rows": rows, "deltas": deltas}

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dtype",
                "eval",
                "score",
                "latency",
                "output_throughput",
                "metrics_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"# {model_label} SSM State Dtype Accuracy\n", "", "## Scores", ""]
    lines.extend(scoring.render_score_table(scores, dtypes, evals))
    lines += ["", "## Deltas", ""]
    lines.extend(scoring.render_delta_table_per_model(deltas))
    lines += ["", "## Artifacts", ""]
    for row in rows:
        lines.append(f"- {row['dtype']} {row['eval']}: `{row['metrics_file']}`")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--model-label", default=os.environ.get("MODEL_LABEL"))
    parser.add_argument("--dtypes", default=os.environ.get("DTYPES", ""))
    parser.add_argument("--evals", default=os.environ.get("EVALS", ""))
    args = parser.parse_args(argv)

    if not args.output_dir or not args.model_label:
        parser.error(
            "--output-dir and --model-label are required (or set OUTPUT_DIR/MODEL_LABEL)"
        )
    output_dir = Path(args.output_dir)
    dtypes = args.dtypes.split()
    evals = args.evals.split()
    if not dtypes or not evals:
        parser.error("--dtypes and --evals must be non-empty")

    rows, scores = collect_rows(output_dir, dtypes, evals)
    write_outputs(output_dir, args.model_label, dtypes, evals, rows, scores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
