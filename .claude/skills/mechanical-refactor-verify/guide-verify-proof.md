# Verify a proof

- How the reviewer of a claimed-mechanical chain (or a single commit) consumes its proof.
- The certified property and primitive contracts: `spec-reproduction-utils.md`; the
  chain-level contract: `spec-reproduction-cli.md`.
- How the proof was produced and the folder it arrives in: `guide-construct-proof.md`.

## 0. Do not trust the PR — verify yourself

- Everything the PR shows you is a **claim**, not evidence: a pasted `PASS` verdict, a
  pasted chain report, a green checkmark, the classification words themselves. All of it
  is text the author (or the author's tooling) produced and could be wrong or fabricated.
- The proof is only ever the run **you** perform locally: run the chain verifier (§1)
  against the PR's actual base and head, with the proof folder you downloaded — never
  approve from the author's pasted output.
- This is cheap by design: the whole point of the machinery is that re-verification is
  one command, so there is no excuse to trust instead of re-run.

## 1. Verify the whole chain

- The default entry point: do not re-run proofs one by one — run the chain verifier:

  ```bash
  python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_reproduction_cli.py \
      --base <base-commit> --branch <pr-branch-name> --proof <folder>
  ```

- It checks every commit declares `mechanical_provable` or `non_mechanical_provable`,
  runs every provable commit's proof, and prints + writes a full report
  (`<folder>/chain_report.md`); exit 0 iff the chain verifies.
- Proofs run up to `--jobs` at a time (default 3; each proof works in its own throwaway
  worktree, so this is safe) — raise it to shorten a long chain's wall clock.
- The contract (word rule, proof resolution, PASS criterion, exit codes):
  `spec-reproduction-cli.md`.
- The `HUMAN_REVIEW` rows in the report are your remaining manual surface — the declared
  non-mechanical commits, plus the §2.3 authored-surface audit of each PASS.
- Each `HUMAN_REVIEW` row carries **two** review duties, and the commit is not approved
  until both hold.
- Duty 1 — **correctness-review the diff itself**: a `non_mechanical_provable` commit is
  exactly the part the machine never certifies, so read its diff and confirm it does
  exactly what its message claims — no lost logic (a branch, a write, an early return
  dropped on the floor), no hidden bug, no unintended behavior change riding along. When
  the commit claims to be behavior-preserving, that means checking equivalence; a commit
  that intentionally changes behavior (a chain need not be a pure refactor) is reviewed
  for the correctness of that change instead. Tests passing is supporting evidence, not
  the review.
- Duty 2 — **verify the declaration itself**: the commit asserts **nothing in it is a
  provable relocation** (`spec-reproduction-cli.md` §2.1), and hiding provable content
  there to dodge the verifier is exactly the escape this chain check exists to close.
    - Read the commit's diff for relocated code. Concretely, run
      `git show <sha> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change`
      and look for moved blocks, and run
      `python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_proof_generator.py <sha>`
      to see what a relocation recipe would cover.
    - A hidden provable part is not a judgement call: demand the split
      (`guide-split.md` §2.2) — do not approve the commit as-is.

## 2. Verify a single commit

### 2.1 Re-run it

- From the repo root:

  ```bash
  python3 <folder>/repro_scripts/<sha>.py
  ```

- When the proof arrived as a gist (`guide-construct-proof.md` §1.3), download it first:

  ```bash
  gh gist clone <gist_id> /tmp/proof        # or: git clone https://gist.github.com/<gist_id>.git /tmp/proof
  cd <repo-root>                            # the run resolves the repo from the cwd
  python3 /tmp/proof/<sha>.py               # PASS = byte-identical to this commit
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
- **UNSUPPORTED** — no recipe inferred (cases: `guide-construct-proof.md` §2.2.2). Not
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
