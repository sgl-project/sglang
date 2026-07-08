---
name: mechanical-refactor-verify
description: Make mechanical refactoring (file splits, function moves, module extractions, renames) machine-checkable instead of eyeballed. Reproduce a relocation commit byte-for-byte from faithful primitives, and split an extraction into a verifiable prepare + move + postpare. Use when doing or reviewing such changes.
---

# Mechanical Refactor — Machine-Checkable Verification

## Overview

The correctness of a mechanical change (file split, function move, module extraction,
rename) must be **machine-checkable, not eyeballed**. Whoever made the change and whenever,
the proof is something anyone can re-run.

There is **one property** — *a commit is a pure relocation* — and **one proof** of it:
**reproduce**. Regenerate the move from the base commit with faithful primitives, run the
formatter, and diff byte-for-byte against the target commit. An empty diff is the proof; any
residual is a bundled non-move change surfaced for review. A reshape that is not a pure
relocation must not ride along: split the work into an optional **prepare**, the certified
**move**, and an optional **postpare** (defined in `spec.md` §2.4; recipes in `guide-prep-move.md`).

## What do you want to do?

- **Do a mechanical refactor** (extract, move, split) → `guide-prep-move.md`: how to cut the
  change into prepare + move + postpare, with the case recipes and anti-patterns.
- **Certify or review a move commit** → `guide-prove.md`: run
  `scripts/mechanical_refactor_generate_proof.py <commit>` (or a `<base>..<tip>` range with
  `--match -move: --out DIR`), audit the emitted script, and hand-write a `Repro` when the
  generator reports `UNSUPPORTED`.
- **Decide whether a change counts as a clean move** → `spec.md`: the property, the whole
  whitelist / not-allowed list, and what a verdict does and does not assert. The single
  source of truth; if any other file disagrees, `spec.md` wins.

## Files

- [`guide-prep-move.md`](guide-prep-move.md) — split a change into prepare + move +
  postpare: the case recipes, what stays mechanical, and the anti-patterns.
- [`guide-prove.md`](guide-prove.md) — prove the move commit: the generator, the `Repro`
  primitives, and how reviewers re-run the proof.
- [`spec.md`](spec.md) — the normative spec of the certified property and its proof.
- [`scripts/mechanical_refactor_generate_proof.py`](scripts/mechanical_refactor_generate_proof.py) —
  the **generator**: infers a reproduce recipe from a commit's diff and emits/runs a
  standalone, auditable script per commit, with a `PASS` / `RESIDUAL` / `UNSUPPORTED` verdict.
- [`scripts/mechanical_refactor_reproduce_utils.py`](scripts/mechanical_refactor_reproduce_utils.py) — the
  **proof engine**: the `Repro` builder's faithful relocation primitives plus the worktree +
  pre-commit + byte-diff scaffold. Self-contained — only git and the standard library.
- [`scripts/tests/`](scripts/tests/) — pytest suites, one folder per module:
  `reproduce_utils/` for the proof engine, `generate_proof/` for the generator.
