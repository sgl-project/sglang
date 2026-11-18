---
name: ci-test-audit
description: Audit the existing test tree and CI configuration for improvements, using a catalog of patterns previously applied in this repo. Use when asked to audit tests or CI, shrink CI time or cost, find redundant or misplaced tests, clean up a test group, or review whether a CI change follows established practice.
---

# CI / Test Audit

Audits what is already in `test/`, `.github/` and `scripts/ci/`. For writing a new test see
[write-sglang-test](../write-sglang-test/SKILL.md); for how the pipeline dispatches and
gates work see [ci-workflow-guide](../ci-workflow-guide/SKILL.md).

## How to use it

Read [action-items.md](action-items.md) first. It is a catalog of patterns that have
been applied to this repo, each with a way to spot it and example PRs. The catalog is the substance of this skill; everything below is only scaffolding.

The user says what to audit -- a test group, a stage, a workflow file, or a complaint
like "this suite is slow" -- and how thoroughly. Read that scope and decide which
patterns apply.

An audit is only useful if it is specific. "This suite could be trimmed" is not a
finding; "`TestFooLargePage` is a strict subset of `TestFooRetractLargePage`, drop it"
is. Every finding needs the file, the pattern it matches, and what to do.

## Auditing at scale

An audit over hundreds of files (a whole tier, every CPU test) splits the reading from
the judgment:

- **Inventory first.** Collect each file's registrations, cost (live estimate x GPUs,
  from the `sglang-ci-stats` model), and the exact items to check (CI-skipped classes,
  overlapping configs) with a script, before any agent reads a file.
- **Fan out file reads to the cheapest model** with a fixed brief and a TSV answer, in
  chunks of about 15 files. The brief lists the exact items to verify; "review every
  case in the file" makes the pass slow and adds nothing.
- **Prosecutor stance.** Each file in the brief is deleted or demoted by default; keeping
  it needs a reason: the change that would turn it red, and the grep that shows no other
  test catches that change.
- **The main agent re-judges** every verdict, most expensive first, before reporting.
  Treat cheap-model "covered by X" or "superseded by Y" verdicts as leads: in one audit
  none of 20 proposed file deletions held on verification.
- **Re-tier before deleting.** For cost, moving a test down a tier (base, extra,
  nightly, weekly) saves far more than deleting: the tier policy is in
  [write-sglang-test, Choosing a Tier](../write-sglang-test/SKILL.md#choosing-a-tier).
  Delete only what never runs in CI or is verified redundant.
- **Failure evidence comes from failed runs' job logs.** `sglang-ci-stats` records only
  successful jobs, so whether a tier catches regressions the tier above misses is read
  from the failed scheduled runs on main.

## Reporting

Group findings by confidence in the claim, not by pattern id. Lead with the ones where
the evidence is in the file you just read; keep the speculative ones separate and say
what you would need to check. Skip anything you are not reasonably sure about -- a long
list of maybes costs more to triage than it saves.

Propose, do not apply. Deleting a test, moving a registration to another stage, and
changing a threshold are all decisions for the user. Land them as separate changes,
since a trim and a threshold change fail for different reasons.

## Keeping the catalog current

Example PRs age. When a pattern shows up in newer work, or a new pattern recurs, update
`action-items.md`: the pattern text should stay general, the examples are
replaceable evidence.
