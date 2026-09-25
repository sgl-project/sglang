#!/usr/bin/env python3
"""Turn a run directory from run_mi45x_local.sh into results.json + report.md.

The registered accuracy tests already report themselves three different ways --
a GitHub step summary table, a `SGLANG_TEST_METRICS_FILE` JSONL of per-file
pass/fail, and stdout -- but none of the three alone is enough. The step
summary has the accuracy numbers but not which file crashed; the metrics JSONL
has per-file status but no accuracy; only stdout distinguishes "server failed
to launch" from "accuracy below threshold". This reads all of them, joins on
model, and emits one artifact per run plus an append-only history file so
successive local runs can be compared before a CI runner exists.

Usable standalone on any past run directory:

    python3 collect_results.py local-ci-results/mi45x/20260910-081500-baseline
    python3 collect_results.py <run-dir> --compare latest
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# `Testing: amd/gpt-oss-120b-w-mxfp4-a-fp8` then, later,
# `  accuracy=0.812 threshold=0.79 ✅ PASS`.
RE_TESTING = re.compile(r"^Testing:\s+(?P<model>\S+)\s*$")
RE_ACCURACY = re.compile(
    r"^\s*accuracy=(?P<acc>[0-9.]+)\s+threshold=(?P<threshold>[0-9.]+)"
)
# run_unittest_files() brackets a JSON-per-file block with these markers.
RE_TIMINGS_BEGIN = re.compile(r"=+ TIMINGS BEGIN =+")
RE_TIMINGS_END = re.compile(r"=+ TIMINGS END =+")
RE_SUITE_MARKER = re.compile(r"^<!--\s*suite:\s*(?P<suite>\S+)\s*-->")
# Markdown table rows are matched structurally and keyed off the header, not by
# a fixed column count: the cookbook tests emit
# `| Model | TP | Accuracy | Cookbook | Threshold | Delta | Status |` while the
# older mi45x tests emit `| Model | TP | Accuracy | Threshold | Status |`, and a
# positional regex silently drops whichever shape it wasn't written for.
RE_TABLE_SEPARATOR = re.compile(r"^\|[\s|:-]+\|$")
# Header cell -> field name. Anything not listed is ignored.
HEADER_ALIASES = {
    "model": "model",
    "tp": "tp",
    "accuracy": "accuracy",
    "cookbook": "cookbook",
    "threshold": "threshold",
    "delta": "delta",
    "status": "status",
}

# Exit codes and their meaning to a human reading the report. run_suite.py
# returns -1 (128-truncated to 255) for "some file failed"; anything else came
# from the shell or a signal.
EXIT_HINTS = {
    0: "no failures reported",
    255: "one or more files failed",
    124: "killed by `timeout`",
    137: "killed (SIGKILL / OOM)",
}


@dataclass
class ModelResult:
    """One model's accuracy measurement, joined across the three sources."""

    suite: str
    model: str
    tp: Optional[int] = None
    accuracy: Optional[float] = None
    threshold: Optional[float] = None
    # The cookbook's measured value, when the test reports one. Distinct from
    # `threshold`, which is that value minus the tolerance.
    cookbook: Optional[float] = None
    status: str = "unknown"


@dataclass
class FileResult:
    """One registered test file, as run_suite.py saw it."""

    suite: str
    test_file: str
    status: str = "unknown"
    duration: Optional[float] = None
    error: Optional[str] = None


@dataclass
class RunResults:
    run_id: str
    run_dir: str
    started_at: Optional[str] = None
    suites: List[str] = field(default_factory=list)
    suite_exit_codes: Dict[str, int] = field(default_factory=dict)
    files: List[FileResult] = field(default_factory=list)
    models: List[ModelResult] = field(default_factory=list)
    env: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            bool(self.files)
            and all(f.status == "pass" for f in self.files)
            and all(rc == 0 for rc in self.suite_exit_codes.values())
        )


def _read_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def parse_suite_status(run_dir: str) -> Dict[str, int]:
    """`<suite>:<rc>` lines written by the shell driver, one per suite."""
    codes: Dict[str, int] = {}
    path = os.path.join(run_dir, "suite_status.txt")
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return codes
    for line in lines:
        suite, _, rc = line.rpartition(":")
        if not suite:
            continue
        try:
            codes[suite] = int(rc)
        except ValueError:
            # "skipped-dry-run" and friends.
            codes[suite] = -1
    return codes


def parse_metrics(run_dir: str) -> Dict[str, dict]:
    """Per-file records flushed by run_unittest_files, keyed by basename.

    The file is shared by every suite in the run and is append-only, so a later
    record for the same basename (a retry, or the same file reached twice) wins.
    """
    records: Dict[str, dict] = {}
    path = os.path.join(run_dir, "metrics.jsonl")
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return records
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "file" and record.get("test_file"):
            records[record["test_file"]] = record
    return records


def parse_suite_log(path: str) -> tuple[List[dict], List[dict]]:
    """Pull (accuracy measurements, TIMINGS entries) out of one suite log."""
    measurements: List[dict] = []
    timings: List[dict] = []
    current_model: Optional[str] = None
    in_timings = False

    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return measurements, timings

    for raw in lines:
        # Logs come through `logger.info`, so most lines carry a prefix; strip
        # it before anchoring on ^ so the patterns work either way.
        line = raw.split("] ", 1)[-1] if "] " in raw else raw

        if RE_TIMINGS_BEGIN.search(raw):
            in_timings = True
            continue
        if RE_TIMINGS_END.search(raw):
            in_timings = False
            continue
        if in_timings:
            start = line.find("{")
            if start != -1:
                try:
                    timings.append(json.loads(line[start:]))
                except json.JSONDecodeError:
                    pass
            continue

        m = RE_TESTING.match(line.strip())
        if m:
            current_model = m.group("model")
            continue

        m = RE_ACCURACY.match(line)
        if m and current_model:
            measurements.append(
                {
                    "model": current_model,
                    "accuracy": float(m.group("acc")),
                    "threshold": float(m.group("threshold")),
                }
            )
            current_model = None

    return measurements, timings


def _split_row(line: str) -> Optional[List[str]]:
    """Cells of a markdown table row, or None if the line is not one."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _as_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_step_summary(run_dir: str) -> Dict[str, List[dict]]:
    """Accuracy tables from the step summary, grouped by the suite marker.

    The driver writes `<!-- suite: NAME -->` before handing the summary to each
    suite; rows after a marker belong to that suite. Column layout is taken from
    each table's own header, so tests with different table shapes both parse.
    """
    by_suite: Dict[str, List[dict]] = {}
    path = os.path.join(run_dir, "step_summary.md")
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return by_suite

    suite = "unknown"
    header: Optional[List[Optional[str]]] = None

    for line in lines:
        marker = RE_SUITE_MARKER.match(line)
        if marker:
            suite = marker.group("suite")
            header = None
            continue

        cells = _split_row(line)
        if cells is None:
            # Prose between tables ends the current table.
            if line.strip():
                header = None
            continue

        if RE_TABLE_SEPARATOR.match(line.strip()):
            continue

        lowered = [c.lower() for c in cells]
        if "model" in lowered and "accuracy" in lowered:
            header = [HEADER_ALIASES.get(c) for c in lowered]
            continue

        if header is None:
            continue

        row: Dict[str, str] = {}
        for field_name, value in zip(header, cells):
            if field_name:
                row[field_name] = value
        if not row.get("model"):
            continue

        tp = row.get("tp", "")
        by_suite.setdefault(suite, []).append(
            {
                "model": row["model"],
                "tp": int(tp) if tp.isdigit() else None,
                "accuracy": _as_float(row.get("accuracy", "")),
                "cookbook": _as_float(row.get("cookbook", "")),
                "threshold": _as_float(row.get("threshold", "")),
                "status": row.get("status", ""),
            }
        )
    return by_suite


def _classify(status_text: str, accuracy: Optional[float], threshold: Optional[float]):
    """Normalise the table's emoji status into pass/fail/error."""
    if "ERROR" in status_text or accuracy is None:
        return "error"
    if "PASS" in status_text:
        return "pass"
    if "FAIL" in status_text:
        return "fail"
    if threshold is None:
        return "unknown"
    return "pass" if accuracy >= threshold else "fail"


def collect(run_dir: str) -> RunResults:
    run_dir = os.path.abspath(run_dir)
    run_meta = _read_json(os.path.join(run_dir, "run.json"))
    env_meta = _read_json(os.path.join(run_dir, "env.json"))

    results = RunResults(
        run_id=run_meta.get("run_id") or os.path.basename(run_dir),
        run_dir=run_dir,
        started_at=run_meta.get("started_at"),
        suites=run_meta.get("suites", []),
        suite_exit_codes=parse_suite_status(run_dir),
        env=env_meta,
    )

    metrics = parse_metrics(run_dir)
    summary_by_suite = parse_step_summary(run_dir)

    log_dir = os.path.join(run_dir, "logs")
    suites = results.suites
    if not suites:
        # No run.json (hand-assembled or truncated run dir): fall back to
        # whatever logs are on disk.
        try:
            names = os.listdir(log_dir)
        except OSError:
            names = []
        suites = sorted(
            os.path.splitext(name)[0] for name in names if name.endswith(".log")
        )

    for suite in suites:
        log_path = os.path.join(log_dir, f"{suite}.log")
        measurements, timings = parse_suite_log(log_path)

        # Prefer the step summary: it is the structured artifact and carries TP
        # and the ERROR rows for models whose server never came up. Stdout is
        # the fallback for a suite killed before the summary was written.
        rows = summary_by_suite.get(suite)
        if rows is None:
            rows = [dict(row, tp=None, status="") for row in measurements]

        for row in rows:
            results.models.append(
                ModelResult(
                    suite=suite,
                    model=row["model"],
                    tp=row.get("tp"),
                    accuracy=row.get("accuracy"),
                    threshold=row.get("threshold"),
                    cookbook=row.get("cookbook"),
                    status=_classify(
                        row.get("status") or "",
                        row.get("accuracy"),
                        row.get("threshold"),
                    ),
                )
            )

        for entry in timings:
            basename = os.path.basename(entry.get("file", ""))
            record = metrics.get(basename, {})
            results.files.append(
                FileResult(
                    suite=suite,
                    test_file=basename,
                    status="pass" if entry.get("passed") else "fail",
                    duration=entry.get("elapsed") or record.get("duration"),
                    error=record.get("error"),
                )
            )

        # A suite that died before printing TIMINGS leaves no file rows at all;
        # without this the report would show a green (empty) suite.
        if not timings and results.suite_exit_codes.get(suite, 0) != 0:
            results.files.append(
                FileResult(
                    suite=suite,
                    test_file="<suite did not report>",
                    status="error",
                    error=f"exit code {results.suite_exit_codes.get(suite)}",
                )
            )

    return results


def render_report(results: RunResults, previous: Optional[RunResults] = None) -> str:
    env = results.env
    git = env.get("git", {})
    rocm = env.get("rocm", {})
    packages = env.get("packages", {})

    lines = [f"# MI455x local accuracy run `{results.run_id}`", ""]
    verdict = "PASS" if results.passed else "FAIL"
    lines += [f"**Overall: {verdict}**", ""]

    lines += ["## Environment", ""]
    lines += [
        f"- host: `{env.get('hostname')}` (container: {env.get('in_container')})",
        f"- commit: `{git.get('describe')}` on `{git.get('branch')}`",
        f"- ROCm: `{rocm.get('version')}`, agents: `{', '.join(rocm.get('agents') or [])}`",
        "- packages: " + ", ".join(f"{k}=`{v}`" for k, v in sorted(packages.items())),
        f"- started: {results.started_at}",
    ]
    dirty = git.get("dirty_files") or []
    if dirty:
        lines.append(
            f"- ⚠️ working tree dirty ({len(dirty)} file(s)); this run is not reproducible"
        )
    lines.append("")

    lines += ["## Accuracy", ""]
    if results.models:
        lines += [
            "| Suite | Model | TP | Accuracy | Cookbook | vs Cookbook | Threshold | Status |",
            "| ----- | ----- | -- | -------- | -------- | ----------- | --------- | ------ |",
        ]
        for m in results.models:
            acc = "N/A" if m.accuracy is None else f"{m.accuracy:.3f}"
            cookbook = "-" if m.cookbook is None else f"{m.cookbook:.3f}"
            threshold = "N/A" if m.threshold is None else f"{m.threshold:.3f}"
            if m.accuracy is None or m.cookbook is None:
                delta = "-"
            else:
                delta = f"{m.accuracy - m.cookbook:+.3f}"
            lines.append(
                f"| {m.suite} | {m.model} | {m.tp or '-'} | {acc} | {cookbook} "
                f"| {delta} | {threshold} | {m.status} |"
            )
    else:
        lines.append("_No accuracy measurements were produced._")
    lines.append("")

    lines += ["## Test files", ""]
    if results.files:
        lines += [
            "| Suite | File | Status | Duration (s) | Error |",
            "| ----- | ---- | ------ | ------------ | ----- |",
        ]
        for f in results.files:
            duration = "-" if f.duration is None else f"{f.duration:.0f}"
            lines.append(
                f"| {f.suite} | {f.test_file} | {f.status} | {duration} "
                f"| {f.error or ''} |"
            )
    else:
        lines.append("_No test files ran._")
    lines.append("")

    if results.suite_exit_codes:
        lines += ["## Suite exit codes", ""]
        for suite, rc in results.suite_exit_codes.items():
            hint = EXIT_HINTS.get(rc, "see log")
            lines.append(f"- `{suite}`: {rc} ({hint})")
        lines.append("")

    if previous is not None:
        lines += [f"## Comparison against `{previous.run_id}`", ""]
        prev_by_key = {(m.suite, m.model): m for m in previous.models}
        rows = []
        for m in results.models:
            prev = prev_by_key.get((m.suite, m.model))
            if prev is None or prev.accuracy is None or m.accuracy is None:
                continue
            delta = m.accuracy - prev.accuracy
            rows.append(
                f"| {m.model} | {prev.accuracy:.3f} | {m.accuracy:.3f} | {delta:+.3f} |"
            )
        if rows:
            lines += [
                "| Model | Previous | Current | Delta |",
                "| ----- | -------- | ------- | ----- |",
            ] + rows
        else:
            lines.append("_No models are comparable between the two runs._")
        lines.append("")

    return "\n".join(lines) + "\n"


def append_history(results: RunResults, history_path: str) -> None:
    """One row per (run, model), so accuracy drift is greppable across runs."""
    fields = [
        "run_id",
        "started_at",
        "commit",
        "dirty",
        "rocm",
        "suite",
        "model",
        "tp",
        "accuracy",
        "cookbook",
        "threshold",
        "status",
    ]
    git = results.env.get("git", {})
    exists = os.path.exists(history_path)
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    with open(history_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for m in results.models:
            writer.writerow(
                {
                    "run_id": results.run_id,
                    "started_at": results.started_at or "",
                    "commit": (git.get("commit") or "")[:12],
                    "dirty": bool(git.get("dirty_files")),
                    "rocm": (results.env.get("rocm") or {}).get("version") or "",
                    "suite": m.suite,
                    "model": m.model,
                    "tp": m.tp if m.tp is not None else "",
                    "accuracy": "" if m.accuracy is None else f"{m.accuracy:.4f}",
                    "cookbook": "" if m.cookbook is None else f"{m.cookbook:.4f}",
                    "threshold": "" if m.threshold is None else f"{m.threshold:.4f}",
                    "status": m.status,
                }
            )


def find_previous_run(run_dir: str) -> Optional[str]:
    """The newest sibling run directory that already has a results.json."""
    parent = os.path.dirname(os.path.abspath(run_dir))
    current = os.path.basename(os.path.abspath(run_dir))
    candidates = []
    try:
        entries = os.listdir(parent)
    except OSError:
        return None
    for name in sorted(entries, reverse=True):
        if name == current:
            continue
        path = os.path.join(parent, name)
        if os.path.exists(os.path.join(path, "results.json")):
            candidates.append(path)
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory from run_mi45x_local.sh.")
    parser.add_argument(
        "--compare",
        metavar="RUN_DIR|latest",
        help="Diff accuracy against another run. 'latest' picks the newest "
        "sibling run that has a results.json.",
    )
    parser.add_argument(
        "--history",
        metavar="CSV",
        help="Append one row per model. (default: <run_dir>/../history.csv)",
    )
    parser.add_argument(
        "--no-history", action="store_true", help="Skip the history CSV."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"error: not a directory: {args.run_dir}", file=sys.stderr)
        return 2

    results = collect(args.run_dir)

    previous = None
    if args.compare:
        compare_dir = args.compare
        if compare_dir == "latest":
            compare_dir = find_previous_run(args.run_dir)
        if compare_dir and os.path.isdir(compare_dir):
            previous = collect(compare_dir)

    report = render_report(results, previous)
    with open(os.path.join(results.run_dir, "report.md"), "w") as f:
        f.write(report)
    with open(os.path.join(results.run_dir, "results.json"), "w") as f:
        json.dump(asdict(results) | {"passed": results.passed}, f, indent=2)

    if not args.no_history:
        history = args.history or os.path.join(
            os.path.dirname(results.run_dir), "history.csv"
        )
        append_history(results, history)

    print(report)
    return 0 if results.passed else 1


if __name__ == "__main__":
    sys.exit(main())
