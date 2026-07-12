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
  (`guide-split.md` §2).
- The rest of the message format is unconstrained **by the machine rule**: the word may
  appear anywhere in the subject or body, in any surrounding syntax. The authoring
  contract additionally fixes the subject format
  (`<group-id>(<commit-id>,<kind>): <message>`, `guide-split.md` §5), which satisfies
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
- A `non_mechanical_provable` commit has no proof obligation, but it is not exempt from
  machine scrutiny: the **mislabel sniff** (§3.5) runs on it. A commit that reproduces
  **fully** as pure relocations is machine-proven to be `mechanical_provable`, so its
  declaration is false: verdict `MISLABELED_PROVABLE`. Otherwise its verdict is
  `HUMAN_REVIEW` — the report marks it for eyes (with a warning when the sniff found
  partial relocations), never certifies it.
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
- `--no-provable-sniff`: skip the mislabel sniff (§3.5).

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

### 3.5 The mislabel sniff

- For every `non_mechanical_provable` commit, the verifier runs the proof generator's
  inference (`mechanical_refactor_proof_generator.py`) on the commit:
    - inference finds no relocation → nothing to sniff, verdict `HUMAN_REVIEW`;
    - the inferred recipe reproduces the commit **byte-for-byte** → the whole commit is a
      pure relocation, the declaration is machine-refuted: verdict `MISLABELED_PROVABLE`
      (relabel it `mechanical_provable` and attach the generated proof);
    - the recipe reproduces with a **residual** → the commit bundles relocations with
      other changes: verdict stays `HUMAN_REVIEW`, but the report carries a **warning**
      naming the inferred relocations — the reviewer decides whether a
      `mechanical_provable` split was dodged (`guide-verify-proof.md` §1).
- The sniff is advisory infrastructure and must never crash the chain walk: a sniff
  error, or a missing generator module, degrades to `HUMAN_REVIEW` with a warning note.
- `--no-provable-sniff` disables it (e.g. for speed on a huge chain); the report then
  carries no mislabel evidence, so only use it when the sniff has already run once.

## 4. The report

- The full report is markdown, printed to stdout **and** written to the report path
  (§3.1), so the folder stays self-describing.
- It contains:
    - the resolved base / branch / proof folder and the **chain verdict**;
    - the commit counts per kind and the proof PASS count;
    - one table row per commit, in chain order: sha, kind, verdict, subject;
    - a **Warnings** section with one entry per ok commit the mislabel sniff flagged
      (§3.5) — partial relocations inside a `non_mechanical_provable` commit;
    - a **Failure details** section with one entry per non-ok commit — the missing-proof
      search locations, the classification rule broken, the mislabel evidence, or the
      failing proof's output tail.
- Verdict vocabulary: `PASS`, `HUMAN_REVIEW`, `FAIL`, `MISSING_PROOF`,
  `AMBIGUOUS_PROOF`, `MISLABELED_PROVABLE`, `UNCLASSIFIED`, `AMBIGUOUS_KIND` — the first
  two are the only ok verdicts.

## 5. Exit codes

- `0` — the chain verifies (§2).
- `1` — the chain was walked but at least one commit does not verify.
- `2` — setup error: unresolvable ref, base not an ancestor, empty range, merge commit in
  the chain, or a missing proof folder. Nothing was certified either way.
