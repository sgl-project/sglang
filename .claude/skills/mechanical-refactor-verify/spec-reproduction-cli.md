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
    - `non_mechanical_provable` — the commit declares it carries changes no relocation
      proof can certify, and is left to human review.
- The rest of the message format is unconstrained: the word may appear anywhere in the
  subject or body, in any surrounding syntax.
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
  `HUMAN_REVIEW` — the report marks it for eyes, never certifies it.
- The chain verdict is PASS iff every commit's verdict is `PASS` or `HUMAN_REVIEW`.

## 3. The CLI contract

### 3.1 Invocation

```bash
python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_reproduction_cli.py \
    --base <base-commit> --branch <pr-branch-name> --proof path/to/proof/folder
```

- `--base` / `--branch`: any commit-ish; both must resolve, `base` must be an ancestor of
  `branch`.
- `--proof`: the proof folder (must exist) — typically the generator's `--out` product
  (`guide-construct-proof.md` §2.2).
- `--repo-root DIR`: run against that repo instead of the cwd's.
- `--report PATH`: write the report there instead of `<proof>/chain_report.md`.

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
  `guide-construct-proof.md` §5.1).
- Proofs are keyed by current shas: after a rebase the shas change, so the proofs must be
  regenerated for the rebased chain.

### 3.4 Proof execution and the PASS criterion

- Each proof runs as `python3 <script>` with the repo root as cwd (the run resolves the
  repo from the cwd, `guide-verify-proof.md` §2.1).
- A proof PASSes iff **both**: exit code 0, **and** the arbiter's `PASS:` verdict line on
  stdout. Requiring the line keeps an old-style script that exits 0 while printing a
  residual from false-passing; requiring the exit code keeps a crash before any verdict
  from passing.

## 4. The report

- The full report is markdown, printed to stdout **and** written to the report path
  (§3.1), so the folder stays self-describing.
- It contains:
    - the resolved base / branch / proof folder and the **chain verdict**;
    - the commit counts per kind and the proof PASS count;
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
