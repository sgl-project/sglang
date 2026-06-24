---
name: mechanical-refactor-verify
description: Make mechanical refactoring (file splits, function moves, module extractions, renames) machine-checkable instead of eyeballed. Reproduce a relocation commit byte-for-byte from faithful primitives, and split an extraction into a verifiable prep + move. Use when doing or reviewing such changes.
user_invocable: true
argument: "[reproduce <base>..<tip>] to certify relocation commits, or omit for the full workflow guide"
---

# Mechanical Refactor — Machine-Checkable Verification

## Core principle

The correctness of a mechanical change (file split, function move, module extraction,
rename) must be **machine-checkable, not eyeballed**. Whoever made the change and
whenever, the proof is something anyone can re-run.

There is **one property** — *a commit is a pure relocation* — and **one proof** of it:
**reproduce**. Regenerate the move from the base commit with faithful AST primitives, run
the formatter, and diff byte-for-byte against the target commit. An empty diff is a machine
proof; any residual is a bundled non-move change surfaced for review.

The proof is a single self-contained script next to this skill —
**`mechanical_refactor_reproduce_utils.py`** — needing only git and the standard library,
so anyone can re-run the result. For a relocation commit,
**`mechanical_refactor_reproduce_gen_utils.py`** infers the recipe and emits the script for
you.

What counts as a clean move — the one property this proof certifies, and exactly how the
reproduce-and-byte-diff establishes it — is specified in **`verifier-spec.md`**, the single
source of truth that the scripts, their tests, and this guide all follow.

A mechanical commit/PR contains **only** mechanical changes (moves, splits, renames,
import fixes, formatting). Semantic changes (new logic, API/signature redesign,
behavior change) go in their own commit/PR.

## Reproduce a relocation commit

You compose a relocation from the `Repro` builder's faithful primitives (`move_symbol`,
`extract_to_new_module`, `lower_call_sites`, `requalify_call_sites`, `remove_import`,
`add_import`, `repath_import`, `add_typechecking_import`) and let the skill regenerate the
commit in a worktree and diff it byte-for-byte against the target. For a relocation commit you do not write this by
hand — `mechanical_refactor_reproduce_gen_utils.py` **infers the recipe and emits the
reproduce script** for you:

```bash
# a range: write a self-contained folder (repro_scripts/<sha>.py + output.log + output.html)
python3 .claude/skills/mechanical-refactor-verify/mechanical_refactor_reproduce_gen_utils.py \
    <base>..<tip> --match -move: --out repro_out

# one commit: print the inferred script and run it
python3 .claude/skills/mechanical-refactor-verify/mechanical_refactor_reproduce_gen_utils.py <commit>
```

Each commit gets a verdict: `PASS` (byte-identical — certified a clean move), `RESIDUAL`
(a non-empty diff — the bundled non-move change to review), or `UNSUPPORTED` (no definition
relocated, e.g. a rename or statement reorder, or an inference gap — review as prep or
hand-write the `Repro`).

→ Full step-by-step, the `Repro` primitives, and the auto-generator: **`reproduce-mode.md`**.

## Make extractions verifiable: split prep + move

A move is certifiable only when its body is byte-identical and its only other changes
are mechanical move artifacts — imports, a dropped `@staticmethod`, requalifying the
moved symbol's call sites, and (for a new module) the scaffolding the prep commit staged
in the source's tail (see `verifier-spec.md`). De-self'ing a method — turning `self.x`
reads into parameters — is behavior-preserving but is a *reshape*, not a move, so it must
not ride along in the move commit. The same goes for a rename or a statement-level reorder.
Split such an extraction into a **prep** commit (the small in-place reshape, no relocation,
checked by tests) followed by a **move** commit (the pure relocation, certified by the
reproduce proof). Prep is the part a human reviews, so keep its diff small.

→ The full philosophy — why, the two-commit recipe, the class-extraction technique, the
new-module technique, what counts as mechanical, and the anti-patterns: **`prep-and-move.md`**.

## Reviewing someone else's PR

- Run the one-click command from the PR description, or
  `mechanical_refactor_reproduce_gen_utils.py <base>..<tip> --match -move: --out DIR`;
  `PASS` means the commit's diff is byte-identical to what the generated script produces.
- A `RESIDUAL` is the exact bundled non-move change to read; an `UNSUPPORTED` commit is a
  rename / reorder / non-move that belongs in prep and gets ordinary review.
