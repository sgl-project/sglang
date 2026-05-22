"""Print a markdown table of lm-eval scores from downloaded eval artifacts.

Reads every `results_*.json` under `<results_dir>` and prints one row per
(config, task) with the scored metrics, threshold (if known), and pass/fail.
Sibling `meta_env.json` files provide the config context.

Usage:
    python3 summarize_evals.py <results_dir>
"""

import json
import sys
from pathlib import Path

from tabulate import tabulate

# Keep in sync with eval-workspace/utils/evals/thresholds.json.
THRESHOLDS = {
    "gsm8k": 0.85,
}


def _load_json(path):
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 summarize_evals.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    rows = []

    for path in results_dir.rglob("results_*.json"):
        if "agg_eval" in path.name or "agg_result" in path.name:
            continue
        data = _load_json(path)
        if not data:
            continue
        # Sibling meta_env.json for context. Fall back to empty.
        meta = _load_json(path.parent / "meta_env.json") or {}

        for task, metrics in (data.get("results") or {}).items():
            scores = {
                name: val
                for name, val in metrics.items()
                if isinstance(val, (int, float)) and "stderr" not in name
            }
            if not scores:
                continue

            # Primary score = first non-stderr metric; threshold check uses it.
            primary = next(iter(scores.values()))
            threshold = THRESHOLDS.get(task)
            status = (
                ("✅" if primary >= threshold else "❌")
                if threshold is not None
                else "—"
            )

            rows.append(
                [
                    meta.get("model_prefix", "?"),
                    meta.get("precision", "?"),
                    f"{meta.get('isl', '?')}x{meta.get('osl', '?')}",
                    task,
                    " · ".join(f"{k}={v:.4f}" for k, v in scores.items()),
                    f"{threshold}" if threshold is not None else "—",
                    status,
                ]
            )

    if not rows:
        print("No eval results found.")
        return

    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    print(
        tabulate(
            rows,
            headers=[
                "Model",
                "Precision",
                "ISLxOSL",
                "Task",
                "Score",
                "Threshold",
                "Status",
            ],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
