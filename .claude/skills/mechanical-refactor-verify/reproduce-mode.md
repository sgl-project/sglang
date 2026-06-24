# Mode A — Reproduce a mechanical PR byte-for-byte

Use when the whole PR is one mechanical refactor, or for a rename / inline where a
formatter re-wraps lines (reproduce-and-diff is more robust there than inspecting the
diff). See `SKILL.md` for when to pick this mode over verify mode.

You write a `transform()` that recreates the change; the skill checks out the base
commit in a throwaway worktree, runs your transform, runs pre-commit, and diffs the
result against the target commit. Empty diff = `PASS`.

## Step 1: Write the transform script

Write it to a scratch path outside the repo (e.g. `/tmp/transform_<short>.py`). The
scaffold (worktree, pre-commit, diff, reporting) lives in the skill's utils — the
script only defines `transform()` and calls `verify_mechanical_refactor`.

```python
#!/usr/bin/env python3
"""Reproducible transform for: <describe the mechanical move>."""
import sys
from pathlib import Path

sys.path.append(".claude/skills/mechanical-refactor-verify")
from reproduce_refactor import (
    verify_mechanical_refactor,
    exec_command,
    git_add_and_commit,
    dedent,
)

BASE_COMMIT = "<base_sha>"
TARGET_COMMIT = "<final_sha_of_the_mechanical_change>"


def transform(dir_root: Path) -> None:
    # Example: split a file by line ranges.
    source = dir_root / "path/to/source.py"
    lines = source.read_text().splitlines(keepends=True)
    for target_path, start, end in [
        ("path/to/pkg/target_a.py", 1, 50),
        ("path/to/pkg/target_b.py", 51, 120),
    ]:
        target = dir_root / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(lines[start - 1 : end]))
    source.unlink()
    git_add_and_commit("mechanical: split source.py", cwd=str(dir_root))
    # A rename is just: for each file, write content.replace(OLD, NEW); commit.
    # pre-commit run --all-files is invoked automatically after transform() returns.


if __name__ == "__main__":
    verify_mechanical_refactor(BASE_COMMIT, TARGET_COMMIT, transform)
```

## Step 2: Run it from the repo root

```bash
python3 /tmp/transform_<short>.py
# Expected: "PASS: transform reproduces the commit exactly."
```

If FAIL, the script prints the non-empty diff. Fix the script and re-run until PASS.

## Step 3: Make the proof re-runnable by reviewers

Put the script where a reviewer can run it (a gist, or attached to the PR
description), and include in the PR description:

````markdown
## Mechanical move — reproducible

```bash
python3 <(curl -sL <raw_url>)   # PASS = byte-identical to this PR
```
````

A mechanical PR contains **only** mechanical changes (moves, splits, renames, import
fixes, formatting). Semantic changes go in a separate PR.
