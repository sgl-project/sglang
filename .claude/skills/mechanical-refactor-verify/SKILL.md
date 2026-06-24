---
name: mechanical-refactor-verify
description: Make mechanical refactoring (file splits, function moves, module extractions) machine-checkable instead of eyeballed. Reproduce a whole mechanical PR byte-for-byte, or certify individual relocation commits inside a mixed stack. Use when doing or reviewing such changes.
user_invocable: true
argument: "[move <commit>] to certify a relocation commit, or omit for the full workflow guide"
---

# Mechanical Refactor — Machine-Checkable Verification

## Core principle

The correctness of a mechanical change (file split, function move, module extraction,
rename) must be **machine-checkable, not eyeballed**. Whoever made the change and
whenever, the proof is something anyone can re-run.

There are two ways to make it checkable. Pick by the shape of the work:

| Your situation | Mode | The proof |
|---|---|---|
| One PR is a single mechanical refactor; or a rename / inline where a formatter re-wraps lines | **Reproduce** | a transform script regenerates the PR's diff byte-for-byte |
| A stack of commits (each its own PR), only some mechanical, mixed with semantic ones | **Verify** | a verifier certifies each mechanical commit is a faithful relocation; semantic commits get ordinary review |

Both modes depend only on `mechanical_refactor_verify_utils.py` next to this skill —
no external scripts or services are required to check the result.

---

## Mode A — Reproduce (one mechanical PR)

Use when the whole PR is mechanical, or for a rename / inline (the formatter handles
re-wrapping, so reproduce-and-diff is more robust than inspecting the diff).

### Step 1: Write the transform script

Write it to a scratch path outside the repo (e.g. `/tmp/transform_<short>.py`). The
scaffold (worktree, pre-commit, diff, reporting) lives in the skill's utils — the
script only defines `transform()` and calls `verify_mechanical_refactor`.

```python
#!/usr/bin/env python3
"""Reproducible transform for: <describe the mechanical move>."""
import sys
from pathlib import Path

sys.path.append(".claude/skills/mechanical-refactor-verify")
from mechanical_refactor_verify_utils import (
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

### Step 2: Run it from the repo root

```bash
python3 /tmp/transform_<short>.py
# Expected: "PASS: transform reproduces the commit exactly."
```

If FAIL, fix the script and re-run until PASS.

### Step 3: Make the proof re-runnable by reviewers

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

---

## Mode B — Verify (a stack of mixed commits)

Use when the work is a chain where each commit becomes its own PR, and only some
commits are mechanical. There is no reproduce script; you certify each mechanical
commit directly.

1. **Classify each commit**: a mechanical *relocation* (function move, file split,
   module extraction) vs a *semantic* change (new logic, API/signature redesign,
   behavior change).

2. **Certify each relocation commit**:

   ```bash
   python3 .claude/skills/mechanical-refactor-verify/mechanical_refactor_verify_utils.py move <commit>
   ```

   It reports, for the commit's diff:
   - how many lines were **relocated byte-for-byte** (indentation ignored, so a
     method becoming an indentation-shifted free function still counts);
   - the **wiring** lines (the new import and the rewritten call sites);
   - any **to review** lines — the only thing a human must read.

   `CLEAN MOVE` means nothing needs review. Otherwise read the (usually tiny)
   to-review set and confirm each line is an equivalent adaptation. Optional eyeball
   cross-check:

   ```bash
   git show <commit> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change
   ```

3. **Semantic commits** get ordinary human review — the verifier does not apply to
   them. The whole point is that mixing mechanical and semantic commits in one branch
   is fine: each commit is reviewed by the method that fits it.

### When an extraction is not a single clean move: split prep + move

De-self'ing a method (turning `self.x` reads into parameters, narrowing a signature)
is behavior-preserving but **not** byte-identical, so it cannot be certified as a pure
move. Split such an extraction into two commits:

- a **prep** commit — the behavior-preserving reshape, done **in place** (de-self,
  narrow the signature, qualify the call site). Reviewed for equivalence by tests /
  lint; it is not byte-mechanical and is meant to be read.
- a **move** commit — the pure relocation of the now-reshaped block to its new home.
  Certified by `move <commit>`.

This isolates the byte-faithful relocation (machine-checked) from the small semantic
reshape (human-checked), so neither has to be taken on trust.

---

## Reviewing someone else's PR

- **Reproduce-mode PR**: run the one-click command from the PR description; `PASS`
  means the diff is byte-identical to what the script produces.
- **Verify-mode PR** (a mechanical commit): run `move <commit>`; confirm `CLEAN MOVE`,
  or that the small to-review set is only equivalent wiring.
