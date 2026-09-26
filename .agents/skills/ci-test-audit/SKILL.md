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
