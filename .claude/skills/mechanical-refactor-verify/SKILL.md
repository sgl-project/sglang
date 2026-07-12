---
name: mechanical-refactor-verify
description: Make mechanical refactoring (file splits, function moves, module extractions, renames) machine-checkable instead of eyeballed. Reproduce a relocation commit byte-for-byte from faithful primitives, and split an extraction into a verifiable prepare + move + postpare. Use when doing or reviewing such changes.
---

# Mechanical Refactor — Machine-Checkable Verification

## 1. Overview

- The correctness of a mechanical change (file split, function move, module extraction,
  rename) must be **machine-checkable, not eyeballed** — the proof is something anyone can
  re-run, whoever made the change and whenever.
- **One property**: *a commit is a pure relocation*. **One proof**: **reproduce** —
  regenerate the move from the base commit with faithful primitives, run the formatter,
  byte-diff against the target.
- Empty diff = the proof. Any residual = a bundled non-move change, surfaced for review.
- A reshape must not ride along: split into optional **prepare** + certified **move** +
  optional **postpare** (`guide-split.md`).

## 2. What do you want to do?

- **Split a change into commits** (extract, move, file split) → `guide-split.md`: the
  prepare + move + postpare rule, the case recipes, and the anti-patterns.
- **Construct the proof for a move commit** → `guide-construct-proof.md`: run
  `scripts/mechanical_refactor_proof_generator.py`, or hand-write a `Repro` when the
  generator reports `UNSUPPORTED`.
- **Verify someone's proof** → `guide-verify-proof.md`: re-run it, read the verdict, audit
  the authored surfaces.
- **Decide whether a change counts as a clean move** → `spec-reproduction-utils.md`: the
  property, the whole whitelist / not-allowed list, and each primitive's contract. The
  source of truth for the reproduction module; if any other file disagrees, it wins.

## 3. Files

- [`guide-split.md`](guide-split.md) — split a change into prepare + move + postpare: the
  case recipes, what stays mechanical, and the anti-patterns.
- [`guide-construct-proof.md`](guide-construct-proof.md) — produce the proof: the
  generator, the hand-written `Repro`, and publishing the proof with the PR.
- [`guide-verify-proof.md`](guide-verify-proof.md) — consume the proof: re-run, verdicts,
  and the audit checklist for authored surfaces.
- [`spec-reproduction-utils.md`](spec-reproduction-utils.md) — the normative spec of the
  clean-move property and the reproduction primitives.
- [`scripts/mechanical_refactor_proof_generator.py`](scripts/mechanical_refactor_proof_generator.py) —
  the **generator**: infers a reproduce recipe from a commit's diff and emits/runs a
  standalone, auditable script per commit, with a `PASS` / `RESIDUAL` / `UNSUPPORTED` verdict.
- [`scripts/mechanical_refactor_reproduction_utils.py`](scripts/mechanical_refactor_reproduction_utils.py) — the
  **proof engine**: the `Repro` builder's faithful relocation primitives plus the worktree +
  pre-commit + byte-diff scaffold. Self-contained — only git and the standard library.
- [`scripts/tests/`](scripts/tests/) — pytest suites, one folder per module:
  `reproduction_utils/` for the proof engine, `proof_generator/` for the generator.
