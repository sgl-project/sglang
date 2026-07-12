# Reproduction CLI — chain verification specification (source of truth)

## 1. Scope

- Source of truth for `scripts/mechanical_refactor_reproduction_cli.py`: the
  **verified-chain property** (§2), the CLI contract (§3), the report (§4), and the exit
  codes (§5).
- The single-commit clean-move property and the proof scripts themselves are specified in
  `spec-reproduction-utils.md`; this file only says how a whole chain of commits is
  checked against a folder of such proofs.
- The CLI, its tests, and the guides defer to this file; on any disagreement, this file
  wins.

## 2. The property — a "verified chain"

> A branch is a **verified chain** over a base iff every commit in `base..branch` is
> **classified** (§2.1) and every `mechanical_provable` commit has exactly one **proof**
> in the proof folder whose run **PASSes** (§2.2).

### 2.1 Classification — the word rule

- Every commit message must contain **exactly one** of the two words:
    - `mechanical_provable` — the commit claims to be a machine-provable relocation;
    - `non_mechanical_provable` — the commit declares that **nothing in it** is
      expressible as the whitelisted relocations of `spec-reproduction-utils.md` §2 — it
      is the minimal unprovable residue, left to human review.
- The declaration is an assertion, not an opt-out: labeling provable content
  `non_mechanical_provable` to dodge the verifier **violates the chain property**, even
  where no machine check catches it. A provable part hiding inside a semantic commit
  belongs in its own `mechanical_provable` commit with a proof
  (`guide-split.md` §2.2).
- The rest of the message format is unconstrained **by the machine rule**: the word may
  appear anywhere in the subject or body, in any surrounding syntax. The authoring
  contract additionally fixes the subject format
  (`<group-id>(<commit-id>,<kind>): <message>`, `guide-split.md` §1.1), which satisfies
  this rule by construction; the verifier deliberately checks only the word, so a chain
  from a different convention still verifies.
- A word counts only standalone: delimited by a non-`[0-9A-Za-z_]` character or the
  message boundary, lowercase, so `non_mechanical_provable` never also counts as the bare
  word, and `xmechanical_provable` counts as neither.
- Repeating the same word is fine; the rule is about **which** of the two is declared:
    - neither word present → `UNCLASSIFIED`;
    - both words present → `AMBIGUOUS_KIND`.

### 2.2 The proof obligation

- Each `mechanical_provable` commit must resolve to exactly one proof script (§3.3);
  none is `MISSING_PROOF`, several is `AMBIGUOUS_PROOF`.
- The proof must run to a PASS (§3.4): the commit reproduces byte-for-byte from its
  parent (`spec-reproduction-utils.md` §4). Anything else is `FAIL`.
- A `non_mechanical_provable` commit has no machine obligation; its verdict is
  `HUMAN_REVIEW` — the report marks it for eyes, never certifies it. Whether its
  declaration is honest is the reviewer's duty to check (`guide-verify-proof.md` §1).
- The chain verdict is PASS iff every commit's verdict is `PASS` or `HUMAN_REVIEW`.

## 3. The CLI contract

### 3.1 Invocation

```bash
python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_reproduction_cli.py \
    --base <base-commit> --branch <pr-branch-name> --proof path/to/proof/folder
```

- The unified entry `scripts/mechanical_refactor.py verify` dispatches here with the
  same arguments; the two forms are equivalent.
- `--base` / `--branch`: any commit-ish; both must resolve, `base` must be an ancestor of
  `branch`.
- `--proof`: the proof folder (must exist) — typically the generator's `--out` product
  (`guide-construct-proof.md` §1.2).
- `--repo-root DIR`: run against that repo instead of the cwd's.
- `--report PATH`: write the report there instead of `<proof>/chain_report.md`.
- `--jobs N`: run up to N proofs concurrently (default 3).
- `--skip-passed`: reuse this machine's own earlier PASS verdicts (§3.5).

### 3.2 The chain

- The commits are `git rev-list --reverse base..branch`, i.e. the whole chain in order.
- The chain must be **linear**: a merge commit anywhere in it is a setup error — per-commit
  proofs are meaningless across a merge.
- An empty range is a setup error, not a trivially-green chain.

### 3.3 Proof resolution

- A commit's proof is a `<sha-prefix>.py` whose stem is lowercase hex, at least 7
  characters, and a prefix of the commit's full sha.
- Searched locations, in order, both always considered: `<proof>/repro_scripts/` (the
  generator layout) and `<proof>/` flat (the gist layout,
  `guide-construct-proof.md` §1.3.1).
- Proofs are keyed by current shas: after a rebase the shas change, so the proofs must be
  regenerated for the rebased chain.

### 3.4 Proof execution and the PASS criterion

- Each proof runs as `python3 <script>` with the repo root as cwd (the run resolves the
  repo from the cwd, `guide-verify-proof.md` §2.1).
- A proof PASSes iff **both**: exit code 0, **and** the arbiter's `PASS:` verdict line on
  stdout. Requiring the line keeps an old-style script that exits 0 while printing a
  residual from false-passing; requiring the exit code keeps a crash before any verdict
  from passing.
- Proofs run **concurrently**, up to `--jobs` at a time (default 3). This is safe because
  each proof works in its own throwaway worktree with a unique branch name and never
  touches the checked-out tree; per-proof verdicts are independent. Classification and
  proof resolution stay sequential (they are cheap), and the report keeps chain order
  regardless of completion order. A completion line (`sha  PASS/FAIL`) is printed as each
  proof finishes, so a long chain shows progress.

### 3.5 The passed-proof cache (`--skip-passed`)

- Purpose: incremental re-verification. Re-running a long chain repeats work for proofs
  whose commit and proof did not change; those earlier PASSes can be reused.
- **What is recorded.** Every run (flag or not) records each proof that PASSed into the
  cache; a FAIL is **never** recorded. An entry's key is the triple:
    - the commit's **full sha** (a rebase changes the sha, so a rebased commit never
      hits);
    - the **sha256 of the proof script's bytes**;
    - the **sha256 of the `mechanical_refactor_reproduction_utils.py` bytes** sitting
      next to the script or one level up (the script's only dependency; `""` when
      absent) — an edited engine invalidates the cache.
- **What is skipped.** Only with `--skip-passed`, and only on an exact triple match, is a
  pending proof skipped: its verdict is `PASS`, marked as reused (a
  `proof <sha>  PASS (cached)` progress line, and a reused count in the report). Any
  mismatch — different sha, edited script, edited utils, no entry — runs the proof
  normally.
- **Where the cache lives — and why that is trust-safe.** The cache file
  (`mechanical_refactor_passed_proofs.json`) sits in the repo's **git common dir**
  (`git rev-parse --git-common-dir`), shared across that repo's worktrees. It is
  machine-local state: it never travels with the proof folder, a gist, or the PR, so
  `--skip-passed` can only ever reuse verdicts **this machine's own runs** produced —
  the do-not-trust-the-PR rule (`guide-verify-proof.md` §0) is not weakened.
- A missing, corrupt, or unreadable cache file is treated as empty; the cache is
  best-effort infrastructure and must never fail the chain walk.

## 4. The report

- The full report is markdown, printed to stdout **and** written to the report path
  (§3.1), so the folder stays self-describing.
- It contains:
    - the resolved base / branch / proof folder and the **chain verdict**;
    - the commit counts per kind and the proof PASS count (plus, when any proof was
      skipped via `--skip-passed`, the reused count);
    - one table row per commit, in chain order: sha, kind, verdict, subject;
    - a **Failure details** section with one entry per non-ok commit — the missing-proof
      search locations, the classification rule broken, or the failing proof's output
      tail.
- Verdict vocabulary: `PASS`, `HUMAN_REVIEW`, `FAIL`, `MISSING_PROOF`,
  `AMBIGUOUS_PROOF`, `UNCLASSIFIED`, `AMBIGUOUS_KIND` — the first two are the only ok
  verdicts.

## 5. Exit codes

- `0` — the chain verifies (§2).
- `1` — the chain was walked but at least one commit does not verify.
- `2` — setup error: unresolvable ref, base not an ancestor, empty range, merge commit in
  the chain, or a missing proof folder. Nothing was certified either way.
