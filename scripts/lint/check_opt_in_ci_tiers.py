#!/usr/bin/env python3
"""Check that scripts/ci/opt_in_ci_tiers.json matches the workflows it names.

The slash command handler labels a PR and re-runs exactly the workflows listed
for a tier. A workflow that gates on a tier label but is missing from the list
is never re-run by /run-extra-ci <hardware>; a listed workflow that gates on another
label, or re-fires on `labeled`, cannot be recovered by a rerun.
A null label is a selection of existing workflows with no extra opt-in gate;
those workflows retain their existing event and path conditions.

pr-gate.yml passes with `opted-in=false` instead of failing on a draft PR or
one without `run-ci` or its tier label, so every job that needs it must skip on
that output; only a `*finish` aggregator may run regardless.
"""

import glob
import json
import re
import sys

import yaml

TIERS_FILE = "scripts/ci/opt_in_ci_tiers.json"
PR_GATE = "./.github/workflows/pr-gate.yml"
# Tier names are slash-command arguments; these spellings mean other commands.
RESERVED_TIER_NAMES = {"base", "cpu", "extra", "failed", "full"}
_TIER_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def gate_labels(workflow: dict) -> set[str]:
    """Opt-in labels a workflow accepts through its pr-gate.yml calls."""
    labels = set()
    for job in (workflow.get("jobs") or {}).values():
        if job.get("uses") == PR_GATE:
            inputs = job.get("with") or {}
            label = inputs.get("require-label")
            if label:
                labels.add(label)
    return labels


def jobs_ignoring_opt_in(workflow: dict) -> list[str]:
    """Jobs that need a pr-gate.yml job but do not check its `opted-in`."""
    jobs = workflow.get("jobs") or {}
    gates = {name for name, job in jobs.items() if job.get("uses") == PR_GATE}
    ignoring = []
    for name, job in jobs.items():
        needs = job.get("needs") or []
        needs = {needs} if isinstance(needs, str) else set(needs)
        if (
            needs & gates
            and not name.endswith("finish")
            and "opted-in" not in str(job.get("if", ""))
        ):
            ignoring.append(name)
    return ignoring


def refires_on_labeled(workflow: dict) -> bool:
    # yaml 1.1 parses the `on:` key as boolean True
    triggers = workflow.get("on", workflow.get(True)) or {}
    if not isinstance(triggers, dict):
        return False
    for event in ("pull_request", "pull_request_target"):
        if "labeled" in ((triggers.get(event) or {}).get("types") or []):
            return True
    return False


def main() -> int:
    with open(TIERS_FILE, encoding="utf-8") as f:
        tiers = json.load(f)

    workflows = {}
    for path in sorted(glob.glob(".github/workflows/*.yml")):
        with open(path, encoding="utf-8") as f:
            workflows[path] = yaml.safe_load(f) or {}

    errors = []
    for name, tier in tiers.items():
        label = tier["label"]
        listed = tier["workflows"]
        if not _TIER_NAME.match(name) or name in RESERVED_TIER_NAMES:
            errors.append(f"tier `{name}`: invalid or reserved tier name")
        if label is not None and label != f"run-ci-{name}":
            errors.append(
                f"tier `{name}`: label must be `run-ci-{name}`, got `{label}`"
            )
        if not listed:
            errors.append(f"tier `{name}`: lists no workflows")

        for path in listed:
            if path not in workflows:
                errors.append(f"tier `{name}`: {path} does not exist")
                continue
            if label is None:
                if gate_labels(workflows[path]):
                    errors.append(
                        f"tier `{name}`: {path} requires an opt-in label, "
                        "but the selection has label: null"
                    )
                continue
            if label not in gate_labels(workflows[path]):
                errors.append(
                    f"tier `{name}`: {path} does not call pr-gate.yml "
                    f"with `require-label: {label}`"
                )
            if refires_on_labeled(workflows[path]):
                errors.append(
                    f"tier `{name}`: {path} triggers on `labeled`; a rerun of "
                    "a run started by an unrelated label stays skipped"
                )

        for path, workflow in workflows.items():
            if label in gate_labels(workflow) and path not in listed:
                errors.append(
                    f"tier `{name}`: {path} gates on `{label}` but is not "
                    "listed, so /run-extra-ci never re-runs it"
                )

    for path, workflow in workflows.items():
        for job in jobs_ignoring_opt_in(workflow):
            errors.append(
                f"{path}: job `{job}` needs pr-gate.yml but does not skip on "
                "its `opted-in == 'false'`, so it runs on drafts and without run-ci"
            )

    if errors:
        print(f"ERROR: {TIERS_FILE} or a pr-gate.yml caller is out of sync:")
        for error in errors:
            print(f"  {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
