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
from mechanical_refactor_reproduce_utils import (
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

## The `Repro` builder — relocation primitives instead of a hand-written transform

For a method/function relocation, compose the transform from faithful primitives instead
of editing line ranges by hand. Each primitive does only a relocation-faithful, AST-driven
edit (it never changes logic), so a byte match after the formatter certifies the commit is
exactly that relocation — and a bundled change surfaces as a residual diff.

```python
from mechanical_refactor_reproduce_utils import Repro

r = Repro(base="<base_sha>", target="<commit>")
# Lower call sites / fix imports BEFORE moving, so a call to a moved method from inside
# another moved method is lowered while still in the source and travels with the body.
r.lower_call_sites("update_weights_from_ipc", "ModelRunner", paths=["a.py", "b.py"])
r.remove_import("a.py", "from x import ModelRunner", in_function="update_weights_from_ipc")
r.add_import("dst.py", "import gc")
r.move_symbol("update_weights_from_ipc", src="a.py", dst="dst.py", into_class="WeightUpdater", dedent=0)
r.run()   # PASS = byte-identical; otherwise prints the residual
```

Primitives: `move_symbol` (cut a def with its decorators, drop `@staticmethod`/`@classmethod`,
dedent, paste at the end of a class or at module level), `lower_call_sites`
(`Owner.m(receiver, rest)` → `receiver.m(rest)`), `requalify_call_sites` (`Owner.m(args)` →
`m(args)`), `remove_import` (function-scoped, all occurrences, with the trailing blank),
`add_import` (the formatter's import sorter places it).

## Auto-generate a reproduce script from a commit

`mechanical_refactor_reproduce_gen_utils.py` infers the recipe from a commit's diff and
before-state AST, then emits and runs a standalone, auditable reproduce script — no one
hand-writes it.

```bash
# one commit: print the inferred script and run it
python3 .claude/skills/mechanical-refactor-verify/mechanical_refactor_reproduce_gen_utils.py <commit>

# a range: write a self-contained folder (repro_scripts/<sha>.py + output.log + output.html)
python3 .claude/skills/mechanical-refactor-verify/mechanical_refactor_reproduce_gen_utils.py \
    <base>..<tip> --match -move: --out repro_out
```

A passing script is the proof; its few primitive calls are what a reviewer audits. The
inference covers a method moved onto an existing class (call sites lowered) and a method
moved to a module-level free function (call sites requalified), with local-import removal
and gained module imports. It reports as unsupported: a new-file extract (per
`prep-and-move.md` the new module's scaffolding belongs in prep, so once the chain is split
the move targets an existing module), a move whose source is already a free function (its
callers cannot be inferred from the qualifier alone), and a move that also renames (not a
pure relocation). Write the `Repro` by hand for those.
