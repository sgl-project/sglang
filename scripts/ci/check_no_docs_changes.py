#!/usr/bin/env python3
"""Reject staged changes under the legacy docs/ tree."""

from __future__ import annotations

import subprocess
import sys

ERROR_MESSAGE = """\
Changes under the legacy docs/ directory are not allowed.

The documentation has been migrated. Please make documentation updates in the
corresponding location under docs_new/ instead.
"""


def staged_paths() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMRDTUXB",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    paths = sys.argv[1:] or staged_paths()
    docs_paths = sorted(
        path for path in paths if path == "docs" or path.startswith("docs/")
    )

    if not docs_paths:
        return 0

    print(ERROR_MESSAGE, file=sys.stderr)
    print("Detected legacy docs/ changes:", file=sys.stderr)
    for path in docs_paths:
        print(f"  - {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
