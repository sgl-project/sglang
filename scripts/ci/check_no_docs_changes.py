#!/usr/bin/env python3
"""Reject newly added or modified files under the legacy docs/ tree.

The legacy Sphinx docs/ tree has been removed; documentation now lives under
docs_new/ (Mintlify). This guard keeps it from being recreated: adding or
editing any docs/ path fails. Deletions (and renames out of docs/) are allowed
so cleanup of any leftover files stays unblocked.
"""

from __future__ import annotations

import subprocess
import sys

ERROR_MESSAGE = """\
The legacy docs/ directory has been removed; adding or editing files there is
not allowed.

Please make documentation updates under docs_new/ instead.
"""


def staged_paths() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--no-renames",
            "--diff-filter=ACM",
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
    print("Detected new or modified legacy docs/ paths:", file=sys.stderr)
    for path in docs_paths:
        print(f"  - {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
