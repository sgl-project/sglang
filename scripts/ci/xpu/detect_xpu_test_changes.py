"""Decide whether changed test files need per-commit XPU CI.

Reads a JSON list of changed paths (dorny/paths-filter `list-files: json`)
from the CHANGED_FILES env var or argv[1], and prints `true` if any of them
carries a non-nightly `register_xpu_ci`, else `false`.
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# ci_register.py is stdlib-only; import it directly (bypassing the sglang
# package __init__, which pulls in torch) so this runs on ubuntu-latest.
sys.path.insert(0, str(REPO_ROOT / "python" / "sglang" / "test" / "ci"))

from ci_register import HWBackend, ut_parse_one_file  # noqa: E402


def xpu_per_commit_files(paths):
    hits = []
    for rel in paths:
        path = REPO_ROOT / rel
        # Deleted files cannot be parsed; the job no longer runs them anyway.
        if path.suffix != ".py" or not path.is_file():
            continue
        try:
            registries, _ = ut_parse_one_file(str(path))
        except (SyntaxError, ValueError) as e:
            # Unparsable test file: run XPU CI so run_suite surfaces the error.
            print(f"::warning::{rel}: {e}", file=sys.stderr)
            hits.append(rel)
            continue
        if any(r.backend == HWBackend.XPU and not r.nightly for r in registries):
            hits.append(rel)
    return hits


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CHANGED_FILES", "")
    paths = json.loads(raw) if raw.strip() else []
    hits = xpu_per_commit_files(paths)
    for rel in hits:
        print(f"XPU per-commit test changed: {rel}", file=sys.stderr)
    print("true" if hits else "false")


if __name__ == "__main__":
    main()
