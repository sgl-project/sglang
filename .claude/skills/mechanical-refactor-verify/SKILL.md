---
name: mechanical-refactor-verify
description: Make mechanical refactoring (file splits, function moves, module extractions, renames) machine-checkable instead of eyeballed. Reproduce a whole mechanical PR byte-for-byte, or certify individual relocation commits inside a mixed stack, and split extractions into a verifiable prep + move. Use when doing or reviewing such changes.
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
| One PR is a single mechanical refactor; or a rename / inline where a formatter re-wraps lines | **Reproduce** (Mode A) | a transform script regenerates the PR's diff byte-for-byte |
| A stack of commits (each its own PR), only some mechanical, mixed with semantic ones | **Verify** (Mode B) | a verifier certifies each mechanical commit is a faithful relocation; semantic commits get ordinary review |

Both modes depend only on `mechanical_refactor_verify_utils.py` next to this skill —
no external scripts or services are required to check the result.

A mechanical commit/PR contains **only** mechanical changes (moves, splits, renames,
import fixes, formatting). Semantic changes (new logic, API/signature redesign,
behavior change) go in their own commit/PR.

## Mode A — Reproduce (one mechanical PR)

Use when the whole PR is one mechanical refactor, or for a rename / inline where the
formatter re-wraps lines (reproduce-and-diff is more robust there than inspecting the
diff). You write a small `transform()` and let the skill regenerate the PR in a
worktree and diff it byte-for-byte against the target commit.

→ Full step-by-step and the script template: **`reproduce-mode.md`** (next to this file).

## Mode B — Verify (a stack of mixed commits)

Use when the work is a chain where each commit becomes its own PR, and only some
commits are mechanical. No reproduce script; you certify each mechanical commit
directly.

1. **Classify each commit**: a mechanical *relocation* (function move, file split,
   module extraction) vs a *semantic* change (new logic, API/signature redesign,
   behavior change).

2. **Certify each relocation commit**:

   ```bash
   python3 .claude/skills/mechanical-refactor-verify/mechanical_refactor_verify_utils.py move <commit>
   ```

   It reports, for the commit's diff:
   - how many lines were **relocated byte-for-byte** (indentation ignored, so a method
     becoming an indentation-shifted free function still counts);
   - the **wiring** lines (the new import and the rewritten call sites);
   - any **to review** lines — the only thing a human must read.

   `CLEAN MOVE` means nothing needs review. Otherwise read the (usually tiny)
   to-review set and confirm each line is an equivalent adaptation. Optional eyeball
   cross-check:

   ```bash
   git show <commit> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change
   ```

3. **Semantic commits** get ordinary human review — the verifier does not apply to
   them. Mixing mechanical and semantic commits in one branch is fine: each commit is
   reviewed by the method that fits it.

> The verifier compares the deleted and added line sets (indentation ignored), so it
> certifies that the body did not change during the move; it does not check line
> order within the moved block. The `--color-moved` cross-check above catches a
> reordered block.

### Make extractions verifiable: split prep + move

De-self'ing a method (turning `self.x` reads into parameters, narrowing a signature)
is behavior-preserving but **not** byte-identical, so it cannot be certified as a pure
move on its own. Split such an extraction into a **prep** commit (the in-place reshape,
checked by tests) followed by a **move** commit (the pure relocation, certified by
`move <commit>`).

→ The full philosophy — why, the two-commit recipe, the class-extraction technique,
what counts as mechanical, and the anti-patterns: **`prep-and-move.md`** (next to this
file).

## Reviewing someone else's PR

- **Reproduce-mode PR**: run the one-click command from the PR description; `PASS`
  means the diff is byte-identical to what the script produces.
- **Verify-mode PR** (a mechanical commit): run `move <commit>`; confirm `CLEAN MOVE`,
  or that the small to-review set is only equivalent wiring.
