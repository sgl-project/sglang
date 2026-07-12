# Verify a proof

- How the reviewer of a claimed-mechanical chain (or a single commit) consumes its proof.
- The certified property and primitive contracts: `spec-reproduction-utils.md`; the
  chain-level contract: `spec-reproduction-cli.md`.
- How the proof was produced and the folder it arrives in: `guide-construct-proof.md`.

## 1. Verify the whole chain

- The default entry point: do not re-run proofs one by one — run the chain verifier:

  ```bash
  python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_reproduction_cli.py \
      --base <base-commit> --branch <pr-branch-name> --proof <folder>
  ```

- It checks every commit declares `mechanical_provable` or `non_mechanical_provable`,
  runs every provable commit's proof, and prints + writes a full report
  (`<folder>/chain_report.md`); exit 0 iff the chain verifies.
- The contract (word rule, proof resolution, PASS criterion, exit codes):
  `spec-reproduction-cli.md`.
- The `HUMAN_REVIEW` rows in the report are your remaining manual surface — the declared
  non-mechanical commits, plus the §2.3 authored-surface audit of each PASS.

## 2. Verify a single commit

### 2.1 Re-run it

- From the repo root:

  ```bash
  python3 <folder>/repro_scripts/<sha>.py
  ```

- The run *is* the proof — it replays the primitives from the base commit and byte-diffs
  against the target in a throwaway worktree.
- The script prints the verdict and exits 0 only on PASS (a residual exits non-zero), so
  a harness can consume the exit code.
- Do not trust a pasted verdict you did not re-run.

### 2.2 Read the verdict

- **PASS** — byte-identical: the commit is exactly the relocations listed in the script,
  nothing else.
- **RESIDUAL** — a non-empty diff: precisely the bundled non-move change. Review it as
  semantic content; a legitimate tail fixup (string-literal module path, doc reference)
  belongs in a postpare commit, not the move.
- **UNSUPPORTED** — no recipe inferred (cases: `guide-construct-proof.md` §2.4). Not
  thereby wrong, but not machine-certified: review by hand as a prepare-style reshape, or
  ask the author for a hand-written `Repro`.

### 2.3 Audit the authored surfaces

- A PASS certifies the relocated bytes; the small **authored** surfaces are reproduced
  from the target and need human eyes.
- In the script, check:
    - the `header=` of `extract_symbols_to_new_module` — the module audits its content
      (imports / docstring / TYPE_CHECKING imports / logger / relocated `drop_assigns`
      copies only); what remains for you: should those assignments move at all?
    - a `leave_delegate=` on `move_symbol` — the forwarding stub is authored code in the
      source file;
    - the `signature=` / `return_text=` / `call=` of `extract_function` — the new
      function's interface is authored; only its body is certified;
    - the `drop_assigns=` list — each named constant leaves the source file.

### 2.4 Know what a PASS does and does not assert

- Requalification / lowering / repath in a script is tied to symbols the same script
  relocates; a consumer-only call or import rewrite (no relocated definition) cannot
  reproduce as a move — it surfaces as a residual.
- Whatever the repo's pre-commit hooks auto-fix is absorbed on both sides
  (`spec-reproduction-utils.md` §4) — the hook set is part of what you trust.
- A PASS judges the **shape of a relocation**, not **intent**: "this commit is exactly
  these relocations", not "this relocation was a good idea". Confirm the commit's subject
  matches what the script actually moves before approving.

### 2.5 Why the mechanism is trustworthy

- It runs the real formatter and compares bytes — no diff-shape heuristic to fool
  (`spec-reproduction-utils.md` §4).
- The proof is the few primitive calls in the script; auditing them (plus §2.3) is the
  whole human surface.
- The folder is self-contained and re-runnable by anyone — a CI step or a reviewer —
  without the skill installed.
