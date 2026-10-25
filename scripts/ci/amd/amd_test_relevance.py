#!/usr/bin/env python3
"""Decide whether a PR's main_package changes can reach an AMD test suite.

The main_package filter in pr-test-amd.yml matches every file under test/, so a
PR that only edits tests for other hardware used to run every AMD GPU stage.
Given the changed main_package files, print "true" if any of them can affect an
AMD run and "false" otherwise.

A changed file is skipped (cannot affect an AMD run) when it is:
- under test/manual/ or a test/registered/<hardware>/ directory for other
  hardware, or
- a test_*.py under test/ that does not call register_amd_ci and that no
  AMD-registered test imports.
Everything else counts, including deleted files, helpers, conftest.py and all
source files, so the check only ever drops pure non-AMD test edits.

Usage: amd_test_relevance.py FILE [FILE ...]   (paths relative to repo root)
       CHANGED_FILES='["a", "b"]' amd_test_relevance.py   (JSON list, no args)
"""

import glob
import json
import os
import re
import sys

OTHER_HW_DIRS = ("npu", "xpu", "musa", "mps", "mlx", "cpu", "gb300")
SKIP_PREFIXES = ("test/manual/",) + tuple(
    f"test/registered/{d}/" for d in OTHER_HW_DIRS
)
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(test_\w+)", re.M)


def _read(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def modules_imported_by_amd_tests() -> set:
    """Names of test_* modules that some AMD-registered test imports."""
    imported = set()
    for path in glob.glob("test/registered/**/*.py", recursive=True):
        src = _read(path)
        if "register_amd_ci" in src:
            imported.update(IMPORT_RE.findall(src))
    return imported


def is_relevant(path: str, amd_imports: set) -> bool:
    if path.startswith(SKIP_PREFIXES):
        return False
    name = os.path.basename(path)
    if not (
        path.startswith("test/") and name.startswith("test_") and name.endswith(".py")
    ):
        return True
    if not os.path.isfile(path):  # deleted or renamed away
        return True
    if "register_amd_ci" in _read(path):
        return True
    return name[: -len(".py")] in amd_imports


def main(paths) -> None:
    if not paths:  # no file list: fail open and run the AMD stages
        print("No changed-file list; treating the PR as AMD-relevant", file=sys.stderr)
        print("true")
        return
    amd_imports = modules_imported_by_amd_tests()
    for path in paths:
        if is_relevant(path, amd_imports):
            print(f"AMD-relevant change: {path}", file=sys.stderr)
            print("true")
            return
    print("No changed file reaches an AMD suite", file=sys.stderr)
    print("false")


if __name__ == "__main__":
    main(sys.argv[1:] or json.loads(os.environ.get("CHANGED_FILES", "[]")))
