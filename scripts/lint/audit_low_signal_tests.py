#!/usr/bin/env python3
"""Find oversized, low-signal unit tests before pruning them.

The audit is deliberately limited to unit-test trees. It does not select
kernel parity, end-to-end, accuracy, performance, or stress coverage.

This is a maintenance-cost screen, not a general test-quality score. A file is
selected when it matches at least one of these signals:

* mock proxy: mock call tracking dominates assertions, or mocks heavily
  outnumber direct SGLang imports;
* structural contract: the test inspects implementation source or its name
  advertises a structural/ownership contract;
* sparse behavior: a >=100-line non-numerical test has at most one assertion
  per 20 lines;
* rapid sparse growth: a >=120-line non-numerical test added since 2026-07-01
  has at most one assertion per 10 lines.

One 29-line wrapper is selected with the benchmark it imports; retaining the
wrapper after removing its shared implementation would leave a collection-time
failure.

Run from the repository root. Use ``--revision`` to audit a base revision after
the selected files have been removed from the working tree.
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import pathlib
import re
import subprocess
from collections.abc import Iterable

UNIT_ROOTS = (
    "test/registered/unit",
    "python/sglang/multimodal_gen/test/unit",
)
RAPID_GROWTH_CUTOFF = "2026-07-01"
DEPENDENT_WRAPPERS = {
    "test/registered/unit/mem_cache/test_rust_unified_radix_cache_bench.py",
}

MOCK_REFERENCE_RE = re.compile(
    r"\bMagicMock\b|\bMock\(|\bpatch\(|\bpatch\.object|"
    r"\bSimpleNamespace\b|\bcreate_autospec\b"
)
SGLANG_IMPORT_RE = re.compile(r"from sglang\.(?:srt|multimodal_gen)")
STRUCTURAL_NAME_RE = re.compile(
    r"(?:ratchet|contract|ownership|surface|coverage|decision|declaration)"
)
NUMERICAL_ASSERTION_MARKERS = (
    "allclose",
    "assert_close",
    "assert_equal",
    "assert_array",
    "assertAlmostEqual",
    "assert_allclose",
)
MOCK_ASSERT_ATTRIBUTES = {
    "assert_any_call",
    "assert_called",
    "assert_called_once",
    "assert_called_once_with",
    "assert_called_with",
    "assert_has_calls",
    "assert_not_called",
    "call_args",
    "call_count",
    "called",
}


@dataclasses.dataclass(frozen=True)
class AuditRow:
    path: str
    added: str
    loc: int
    cases: int
    mock_refs: int
    sglang_imports: int
    assertions: int
    call_assertions: int
    numerical: bool
    reasons: tuple[str, ...]

    @property
    def selected(self) -> bool:
        return bool(self.reasons)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout


def _is_unit_test(path: str) -> bool:
    return (
        path.endswith(".py")
        and pathlib.PurePosixPath(path).name.startswith("test_")
        and any(path.startswith(f"{root}/") for root in UNIT_ROOTS)
    )


def _sources(root: pathlib.Path, revision: str | None) -> dict[str, str]:
    if revision:
        paths = _git("ls-tree", "-r", "--name-only", revision, "--", *UNIT_ROOTS)
        return {
            path: _git("show", f"{revision}:{path}")
            for path in paths.splitlines()
            if _is_unit_test(path)
        }

    sources = {}
    for unit_root in UNIT_ROOTS:
        for path in sorted((root / unit_root).rglob("test_*.py")):
            relative = path.relative_to(root).as_posix()
            sources[relative] = path.read_text(encoding="utf-8", errors="replace")
    return sources


def _addition_dates(revision: str | None) -> dict[str, str]:
    args = [
        "log",
        "--diff-filter=A",
        "--format=@@%as",
        "--name-only",
    ]
    if revision:
        args.append(revision)
    args.extend(["--", *UNIT_ROOTS])
    output = _git(*args)

    current_date = ""
    dates: dict[str, str] = {}
    for line in output.splitlines():
        if line.startswith("@@"):
            current_date = line[2:]
        elif _is_unit_test(line):
            # git log is newest first. Overwriting retains the oldest addition
            # if a path was deleted and later recreated.
            dates[line] = current_date
    return dates


def audit_source(path: str, source: str, added: str = "") -> AuditRow | None:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return None

    cases = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
        for node in ast.walk(tree)
    )
    if not cases and path not in DEPENDENT_WRAPPERS:
        return None

    assertions = 0
    call_assertions = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            assertions += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attribute = node.func.attr
            if attribute in MOCK_ASSERT_ATTRIBUTES:
                call_assertions += 1
            elif attribute.startswith("assert"):
                assertions += 1
        elif isinstance(node, ast.Attribute) and node.attr in {
            "call_args",
            "call_count",
            "called",
        }:
            call_assertions += 1

    loc = source.count("\n") + 1
    mock_refs = len(MOCK_REFERENCE_RE.findall(source))
    sglang_imports = len(SGLANG_IMPORT_RE.findall(source))
    numerical = any(marker in source for marker in NUMERICAL_ASSERTION_MARKERS)
    total_assertions = assertions + call_assertions

    reasons = []
    if (assertions > 0 and call_assertions >= assertions) or (
        mock_refs >= 10 and mock_refs >= 3 * max(sglang_imports, 1)
    ):
        reasons.append("mock-proxy")
    if (
        "inspect.getsource" in source
        or "getsource(" in source
        or (STRUCTURAL_NAME_RE.search(pathlib.PurePosixPath(path).name))
    ):
        reasons.append("structural-contract")
    if loc >= 100 and total_assertions * 20 <= loc and not numerical:
        reasons.append("sparse-behavior")
    if (
        added >= RAPID_GROWTH_CUTOFF
        and loc >= 120
        and total_assertions * 10 <= loc
        and not numerical
    ):
        reasons.append("rapid-sparse-growth")
    if path in DEPENDENT_WRAPPERS:
        reasons.append("dependent-wrapper")

    return AuditRow(
        path=path,
        added=added,
        loc=loc,
        cases=cases,
        mock_refs=mock_refs,
        sglang_imports=sglang_imports,
        assertions=assertions,
        call_assertions=call_assertions,
        numerical=numerical,
        reasons=tuple(reasons),
    )


def scan(root: pathlib.Path, revision: str | None = None) -> list[AuditRow]:
    dates = _addition_dates(revision)
    rows = []
    for path, source in _sources(root, revision).items():
        row = audit_source(path, source, dates.get(path, ""))
        if row is not None:
            rows.append(row)
    return rows


def _totals(rows: Iterable[AuditRow]) -> tuple[int, int, int]:
    materialized = list(rows)
    return (
        len(materialized),
        sum(row.loc for row in materialized),
        sum(row.cases for row in materialized),
    )


def write_csv(path: pathlib.Path, rows: Iterable[AuditRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            [
                "file",
                "added",
                "loc",
                "cases",
                "mock_refs",
                "sglang_imports",
                "assertions",
                "call_assertions",
                "reasons",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.path,
                    row.added,
                    row.loc,
                    row.cases,
                    row.mock_refs,
                    row.sglang_imports,
                    row.assertions,
                    row.call_assertions,
                    ";".join(row.reasons),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--revision")
    parser.add_argument("--csv", type=pathlib.Path)
    parser.add_argument("--paths-only", action="store_true")
    args = parser.parse_args()

    selected = sorted(
        (row for row in scan(args.root.resolve(), args.revision) if row.selected),
        key=lambda row: row.path,
    )
    if args.paths_only:
        for row in selected:
            print(row.path)
        return 0

    print("signal                     files       LOC     cases")
    for reason in (
        "mock-proxy",
        "structural-contract",
        "sparse-behavior",
        "rapid-sparse-growth",
        "dependent-wrapper",
    ):
        rows = [row for row in selected if reason in row.reasons]
        file_count, loc, cases = _totals(rows)
        print(f"{reason:26s} {file_count:5d} {loc:9d} {cases:9d}")
    file_count, loc, cases = _totals(selected)
    print(f"{'union':26s} {file_count:5d} {loc:9d} {cases:9d}")

    if args.csv:
        write_csv(args.csv, selected)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
