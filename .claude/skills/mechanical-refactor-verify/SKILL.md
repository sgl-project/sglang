---
name: mechanical-refactor-verify
description: Make mechanical refactoring (file splits, function moves, module extractions, renames) machine-checkable instead of eyeballed. Reproduce a relocation commit byte-for-byte from faithful primitives, and split an extraction into a verifiable prep + move. Use when doing or reviewing such changes.
user_invocable: true
argument: "[reproduce <base>..<tip>] to certify relocation commits, or omit for the full workflow guide"
---

# Mechanical Refactor — Machine-Checkable Verification

## Overview

The correctness of a mechanical change (file split, function move, module extraction,
rename) must be **machine-checkable, not eyeballed**. Whoever made the change and whenever,
the proof is something anyone can re-run.

There is **one property** — *a commit is a pure relocation* — and **one proof** of it:
**reproduce**. Regenerate the move from the base commit with faithful AST primitives, run the
formatter, and diff byte-for-byte against the target commit. An empty diff is the proof; any
residual is a bundled non-move change surfaced for review. For a relocation commit you do not
write the script by hand — the generator infers the recipe and emits it.

A move is certifiable only when its body is byte-identical and its only other changes are
mechanical move artifacts. A reshape that is not a pure relocation — de-self'ing a method, a
rename, a statement reorder — must not ride along; split it into a **prep** commit (the small
human-reviewed reshape) followed by a **move** commit (the pure relocation, certified by the
reproduce proof). A mechanical commit/PR contains **only** mechanical changes; semantic changes
(new logic, API/signature redesign, behavior change) go in their own commit/PR.

## Files

### Read these to use the skill

- [`how-to-guide.md`](how-to-guide.md) — how to **certify a relocation commit**: the auto-generator
  command, what its inference covers, the `UNSUPPORTED` cases, and the `Repro` primitives for
  hand-writing a transform when inference falls short.
- [`mental-model-prep-and-move.md`](mental-model-prep-and-move.md) — how to **make an extraction verifiable** by splitting
  it into prep + move: the two-commit recipe, the class-extraction and new-module
  (trailing-block) techniques, what counts as mechanical, and the anti-patterns.

### Read these only when you need the internals

- [`verifier-spec.md`](verifier-spec.md) — the single **source of truth**: the one property
  certified, what counts as a clean move (the whitelist and the not-allowed list), and exactly
  how reproduce-and-byte-diff establishes it. The code, tests, and other guides all implement
  this file; if any disagrees, this file wins.
- [`scripts/mechanical_refactor_generate_proof.py`](scripts/mechanical_refactor_generate_proof.py) —
  the **generator**: infers a reproduce recipe from a commit's diff and emits/runs a standalone,
  auditable script per commit, with a `PASS` / `RESIDUAL` / `UNSUPPORTED` verdict.
- [`scripts/mechanical_refactor_reproduce_utils.py`](scripts/mechanical_refactor_reproduce_utils.py) — the
  **proof engine**: the `Repro` builder's faithful relocation primitives plus the worktree +
  pre-commit + byte-diff scaffold. Self-contained — only git and the standard library.
- [`scripts/test_mechanical_refactor_generate_proof.py`](scripts/test_mechanical_refactor_generate_proof.py)
  and [`scripts/test_mechanical_refactor_reproduce_utils.py`](scripts/test_mechanical_refactor_reproduce_utils.py)
  — the tests for the generator and the proof engine.
