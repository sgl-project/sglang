#!/usr/bin/env python3
"""Refresh est_time literals from sglang-ci-stats/model.json.

Usage:
    python scripts/ci/update_est_time.py [--dry-run] \\
        [--model-url URL] [--summary-file PATH]
"""

import argparse
import ast
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/sgl-project/sglang-ci-stats/main/model.json"
)

# AMD / NPU live in separate workflows and are not scraped by sglang-ci-stats.
BACKENDS = ("cuda", "cpu")
REGISTER_FUNCS = {f"register_{backend}_ci" for backend in BACKENDS}
# Positional slots of register_*_ci, as in ci_register._PARAM_ORDER.
POSITIONAL_ARGS = ("est_time", "suite")

# A change is "significant" if |delta| >= this many seconds AND the relative
# change is at least SIGNIFICANT_REL_DELTA. Dual threshold filters out both
# tiny absolute drifts on long tests and small-but-noisy relative swings on
# short tests.
SIGNIFICANT_ABS_DELTA = 30
SIGNIFICANT_REL_DELTA = 0.3


def fetch_model(url):
    """Curl model.json. Fail loudly on network or parse errors -- the
    weekly workflow will surface the failure rather than silently making
    a no-op PR."""
    out = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--max-time", "30", url],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def find_est_time_literals(content):
    """Map each suite registered by a module-level `register_{backend}_ci(...)`
    call to the AST node of that call's `est_time` literal.

    Reads the call the same way ci_register.py does, so black-wrapped calls,
    positional `register_cpu_ci(5, "suite")`, float values and any keyword
    order all resolve.
    """
    literals = {}
    for stmt in ast.parse(content).body:
        call = stmt.value if isinstance(stmt, ast.Expr) else None
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in REGISTER_FUNCS
        ):
            continue
        args = dict(zip(POSITIONAL_ARGS, call.args))
        args.update((kw.arg, kw.value) for kw in call.keywords)
        values = {k: v.value for k, v in args.items() if isinstance(v, ast.Constant)}
        suite = values.get("suite") or (
            f"{values.get('stage')}-test-{values.get('runner_config')}"
        )
        if isinstance(args.get("est_time"), ast.Constant):
            literals[suite] = args["est_time"]
    return literals


def update_files(model, dry_run=False):
    """Walk `model.est`, apply each p90 to the matching register call.

    Returns list of (relpath, suite, old, new) for every changed entry.
    """
    by_file = defaultdict(list)
    for suite, files in model.get("est", {}).items():
        for relpath, p90 in files.items():
            by_file[relpath].append((suite, p90))

    changes = []
    for relpath, entries in sorted(by_file.items()):
        filepath = REPO_ROOT / relpath
        if not filepath.exists():
            continue
        content = filepath.read_text()
        literals = find_est_time_literals(content)
        edits = []
        for suite, p90 in entries:
            node = literals.get(suite)
            if node is None or node.value == p90:
                continue
            edits.append((node, p90))
            changes.append((relpath, suite, node.value, p90))
            print(
                f"  {relpath}: suite={suite!r} est_time {node.value} -> {p90}",
                file=sys.stderr,
            )

        if edits and not dry_run:
            # AST columns are UTF-8 byte offsets; edit back to front so earlier
            # offsets on the same line stay valid.
            lines = content.encode().splitlines(keepends=True)
            for node, p90 in sorted(
                edits, key=lambda e: (e[0].lineno, e[0].col_offset), reverse=True
            ):
                line = lines[node.lineno - 1]
                lines[node.lineno - 1] = (
                    line[: node.col_offset]
                    + str(p90).encode()
                    + line[node.end_col_offset :]
                )
            filepath.write_bytes(b"".join(lines))

    return changes


def is_significant(old, new):
    delta = abs(new - old)
    return (
        delta >= SIGNIFICANT_ABS_DELTA and delta / max(old, 1) >= SIGNIFICANT_REL_DELTA
    )


def write_summary(changes, summary_file):
    """Write a markdown summary of significant est_time changes."""
    sig = [c for c in changes if is_significant(c[2], c[3])]
    sig.sort(key=lambda c: abs(c[3] - c[2]), reverse=True)

    lines = []
    if sig:
        lines.append(
            f"### Significant est_time changes ({len(sig)} of {len(changes)} updates)"
        )
        lines.append("")
        lines.append("| File | Suite | Old (s) | New (s) | Δ |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for relpath, suite, old, new in sig:
            delta = new - old
            sign = "+" if delta > 0 else ""
            pct = round(delta / max(old, 1) * 100)
            lines.append(
                f"| `{Path(relpath).name}` | `{suite}` | "
                f"{old} | {new} | {sign}{delta} ({sign}{pct}%) |"
            )
    else:
        lines.append(
            f"_{len(changes)} est_time update(s); none exceeded both "
            f"±{SIGNIFICANT_ABS_DELTA}s and "
            f"±{int(SIGNIFICANT_REL_DELTA * 100)}% thresholds._"
        )

    Path(summary_file).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-url",
        default=DEFAULT_MODEL_URL,
        help="URL of model.json from sglang-ci-stats (file:// is OK for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without modifying files",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Write a markdown summary of significant changes to this path",
    )
    args = parser.parse_args()

    print(f"Fetching {args.model_url}", file=sys.stderr)
    model = fetch_model(args.model_url)
    print(
        f"  model fit_window_start={model.get('fit_window_start')} "
        f"fit_window_days={model.get('fit_window_days')} "
        f"n_runs={model.get('n_runs')} "
        f"n_suites={len(model.get('est', {}))}",
        file=sys.stderr,
    )

    changes = update_files(model, dry_run=args.dry_run)

    n_files = len({c[0] for c in changes})
    action = "Would update" if args.dry_run else "Updated"
    print(
        f"\n{action} {len(changes)} est_time entries across {n_files} files",
        file=sys.stderr,
    )

    if args.summary_file:
        write_summary(changes, args.summary_file)
        print(f"Wrote summary to {args.summary_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
