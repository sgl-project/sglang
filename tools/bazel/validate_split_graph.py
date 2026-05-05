#!/usr/bin/env python3
"""Validate the generated first-party Bazel split used by registered tests."""

from __future__ import annotations

import sys

from gen_source_graph import generated_outputs


def _validate_generated_files(errors: list[str]) -> None:
    for path, expected in generated_outputs().items():
        actual = path.read_text() if path.exists() else ""
        if actual != expected:
            errors.append(f"{path}: generated BUILD file is out of date")
        if "//python:sglang" in actual:
            errors.append(f"{path}: generated BUILD file depends on //python:sglang")


def main() -> int:
    errors: list[str] = []
    _validate_generated_files(errors)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
