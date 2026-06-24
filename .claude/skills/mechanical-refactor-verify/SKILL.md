---
name: mechanical-refactor-verify
description: Make mechanical refactoring (file splits, function moves, module extractions, renames) machine-checkable instead of eyeballed. Reproduce a whole mechanical PR byte-for-byte, or certify individual relocation commits inside a mixed stack, and split extractions into a verifiable prep + move. Use when doing or reviewing such changes.
user_invocable: true
argument: "[verify <commit>] to certify a relocation commit, or omit for the full workflow guide"
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
| A stack of commits (each its own PR), only some mechanical, mixed with semantic ones | **Verify** (Mode B) | a verifier certifies each mechanical commit is an in-order relocation (uniform indent shift allowed) plus only mechanical move artifacts; semantic commits get ordinary review |

Each mode is a single self-contained script next to this skill — **`mechanical_refactor_reproduce_utils.py`**
for Mode A and **`mechanical_refactor_verify_utils.py`** for Mode B — needing only git and the standard
library, so anyone can re-run the result.

The exact rule Mode B's verifier enforces — what counts as a move and what does not —
is specified in **`verifier-spec.md`**, the source of truth that the script, its tests,
and this guide all follow.

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

Use when the work is a chain where each commit becomes its own PR and only some
commits are mechanical. No reproduce script; you classify each commit and certify each
relocation from its diff with
`mechanical_refactor_verify_utils.py <commit>` (`CLEAN MOVE` = every changed line is either part of the moved
block — relocated in order, allowing one uniform indent shift — or a mechanical move
artifact: an import, a dropped `@staticmethod`, or a requalified call site; otherwise it
lists the lines to review). Semantic commits get ordinary review.

→ Full step-by-step, and what the report does and does not assert: **`verification-mode.md`** (next to this file).

## Make extractions verifiable: split prep + move

A move is certifiable only when its body is byte-identical and its only other changes
are mechanical move artifacts — imports, a dropped `@staticmethod`, and requalifying the
moved symbol's call sites (see `verifier-spec.md`). De-self'ing a method — turning
`self.x` reads into parameters — is behavior-preserving but is a *reshape*, not a move,
so it must not ride along in the move commit. Split such an extraction into a **prep**
commit (the small in-place reshape, no relocation, checked by tests) followed by a
**move** commit (the pure relocation, certified by `mechanical_refactor_verify_utils.py <commit>`). Prep is
the part a human reviews, so keep its diff small.

→ The full philosophy — why, the two-commit recipe, the class-extraction technique,
what counts as mechanical, and the anti-patterns: **`prep-and-move.md`** (next to this
file).

## Reviewing someone else's PR

- **Reproduce-mode PR**: run the one-click command from the PR description; `PASS`
  means the diff is byte-identical to what the script produces.
- **Verify-mode PR** (a mechanical commit): run `mechanical_refactor_verify_utils.py <commit>`; confirm
  `CLEAN MOVE`, or that the small to-review set is only equivalent wiring.
