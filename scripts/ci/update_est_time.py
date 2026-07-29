#!/usr/bin/env python3
"""Refresh est_time literals from sglang-ci-stats/model.json.

Usage:
    python scripts/ci/update_est_time.py [--dry-run] \\
        [--model-url URL] [--summary-file PATH]
"""

import argparse
import json
import re
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


def make_patterns(suite):
    """Yield regex objects that match `register_{backend}_ci(est_time=N, ...)`
    for the given suite, covering both registration styles:

      legacy: register_X_ci(est_time=N, suite="<full-suite>")
      new:    register_X_ci(est_time=N, stage="<stage>", runner_config="<rc>")
    """
    stage_rc = None
    if "-test-" in suite:
        stage, _, rc = suite.partition("-test-")
        stage_rc = (stage, rc)
    for backend in BACKENDS:
        yield re.compile(
            rf"(register_{backend}_ci\(est_time=)(\d+)"
            rf'(,\s*suite="{re.escape(suite)}")'
        )
        if stage_rc is not None:
            stage, rc = stage_rc
            yield re.compile(
                rf"(register_{backend}_ci\(est_time=)(\d+)"
                rf'(,\s*stage="{re.escape(stage)}",\s*runner_config="{re.escape(rc)}")'
            )


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
        new_content = content

        for suite, p90 in entries:
            for pattern in make_patterns(suite):
                match = pattern.search(new_content)
                if match is None:
                    continue
                old_val = int(match.group(2))
                if old_val != p90:
                    new_content = pattern.sub(rf"\g<1>{p90}\3", new_content)
                    changes.append((relpath, suite, old_val, p90))
                    print(
                        f"  {relpath}: suite={suite!r} " f"est_time {old_val} -> {p90}",
                        file=sys.stderr,
                    )
                break  # one (file, suite) -> at most one register call

        if new_content != content and not dry_run:
            filepath.write_text(new_content)

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
            f"### Significant est_time changes "
            f"({len(sig)} of {len(changes)} updates)"
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
        f"  model data_as_of={model.get('data_as_of')} "
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
