#!/usr/bin/env python3
"""Screen SGLang tests for mock-driven, low-signal files.

This is a triage tool, not a deletion oracle. It scans the two test trees that
RFC #37405 covers and emits the files matching either of these deliberately
broad signals:

* B: mock call-tracking assertions are at least as numerous as real assertions.
* C: at least ten mock references and at least three mock references per real
  ``sglang.srt`` or ``sglang.multimodal_gen`` import.

Run from the repository root:

    python3 scripts/lint/audit_test_value.py
    python3 scripts/lint/audit_test_value.py --csv /tmp/test-value-candidates.csv

The audit uses only the Python standard library and never imports SGLang.
Every candidate still requires human review: mocks can be the right way to
exercise orchestration, error handling, and hardware-independent control flow.
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import pathlib
import re
from collections.abc import Iterable

TEST_ROOTS = (
    pathlib.Path("test/registered"),
    pathlib.Path("python/sglang/multimodal_gen/test"),
)

MOCK_REFERENCE_RE = re.compile(
    r"\bMagicMock\b|\bMock\(|\bpatch\(|\bpatch\.object|"
    r"\bSimpleNamespace\b|\bcreate_autospec\b"
)
SGLANG_IMPORT_RE = re.compile(r"from sglang\.(?:srt|multimodal_gen)")

MOCK_CALL_ASSERT_ATTRIBUTES = {
    "assert_any_call",
    "assert_called",
    "assert_called_once",
    "assert_called_once_with",
    "assert_called_with",
    "assert_has_calls",
    "assert_not_called",
    "call_count",
    "called",
    "call_args",
}
MOCK_CALL_INSPECTION_ATTRIBUTES = {"call_args", "call_count", "called"}


@dataclasses.dataclass(frozen=True)
class AuditRow:
    path: str
    tree: str
    loc: int
    cases: int
    mock_refs: int
    sglang_imports: int
    real_assertions: int
    call_assertions: int

    @property
    def signal_b(self) -> bool:
        return self.real_assertions > 0 and self.call_assertions >= self.real_assertions

    @property
    def signal_c(self) -> bool:
        return self.mock_refs >= 10 and self.mock_refs >= 3 * max(
            self.sglang_imports, 1
        )

    @property
    def is_candidate(self) -> bool:
        return self.signal_b or self.signal_c


def _tree_name(path: pathlib.Path) -> str:
    normalized = path.as_posix()
    if normalized.startswith("test/registered/"):
        return "test/registered"
    return "multimodal_gen"


def audit_source(path: pathlib.Path, source: str) -> AuditRow | None:
    """Return one audit row, or ``None`` for a file with no test cases."""

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return None

    test_cases = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    ]
    if not test_cases:
        return None

    call_assertions = 0
    real_assertions = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            real_assertions += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attribute = node.func.attr
            if attribute in MOCK_CALL_ASSERT_ATTRIBUTES:
                call_assertions += 1
            elif attribute.startswith("assert"):
                real_assertions += 1
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in MOCK_CALL_INSPECTION_ATTRIBUTES
        ):
            call_assertions += 1

    return AuditRow(
        path=path.as_posix(),
        tree=_tree_name(path),
        loc=source.count("\n") + 1,
        cases=len(test_cases),
        mock_refs=len(MOCK_REFERENCE_RE.findall(source)),
        sglang_imports=len(SGLANG_IMPORT_RE.findall(source)),
        real_assertions=real_assertions,
        call_assertions=call_assertions,
    )


def scan(root: pathlib.Path) -> list[AuditRow]:
    rows = []
    for test_root in TEST_ROOTS:
        absolute_root = root / test_root
        if not absolute_root.is_dir():
            continue
        for path in sorted(absolute_root.rglob("*.py")):
            if path.name in {"__init__.py", "conftest.py"}:
                continue
            if not (path.name.startswith("test_") or path.name.endswith("_test.py")):
                continue
            relative_path = path.relative_to(root)
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            row = audit_source(relative_path, source)
            if row is not None:
                rows.append(row)
    return rows


def candidates(rows: Iterable[AuditRow]) -> list[AuditRow]:
    return sorted((row for row in rows if row.is_candidate), key=lambda row: row.path)


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
                "tree",
                "loc",
                "cases",
                "mock_refs",
                "sglang_imports",
                "real_assertions",
                "call_assertions",
                "signal_b",
                "signal_c",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.path,
                    row.tree,
                    row.loc,
                    row.cases,
                    row.mock_refs,
                    row.sglang_imports,
                    row.real_assertions,
                    row.call_assertions,
                    int(row.signal_b),
                    int(row.signal_c),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--csv", type=pathlib.Path)
    args = parser.parse_args()

    selected = candidates(scan(args.root.resolve()))
    signal_b = [row for row in selected if row.signal_b]
    signal_c = [row for row in selected if row.signal_c]
    registered = [row for row in selected if row.tree == "test/registered"]
    diffusion = [row for row in selected if row.tree == "multimodal_gen"]

    print("signal                              files      LOC    cases")
    for label, rows in (
        ("B: call assertions >= real", signal_b),
        ("C: mock-heavy", signal_c),
        ("B union C", selected),
        ("  test/registered", registered),
        ("  multimodal_gen", diffusion),
    ):
        file_count, loc, case_count = _totals(rows)
        print(f"{label:34s} {file_count:5d} {loc:8d} {case_count:8d}")

    if args.csv is not None:
        write_csv(args.csv, selected)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
