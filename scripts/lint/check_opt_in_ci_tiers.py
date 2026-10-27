#!/usr/bin/env python3
"""Check that scripts/ci/opt_in_ci_tiers.json matches the workflows it names.

The slash command handler labels a PR and re-runs exactly the workflows listed
for a tier. A workflow that gates on a tier label but is missing from the list
is never re-run by /run-<tier>-ci; a listed workflow that gates on another
label, or re-fires on `labeled`, cannot be recovered by a rerun.
A null label is a selection of existing workflows with no extra opt-in gate;
those workflows retain their existing event and path conditions.
"""

import glob
import json
import re
import sys

import yaml

TIERS_FILE = "scripts/ci/opt_in_ci_tiers.json"
PR_GATE = "./.github/workflows/pr-gate.yml"
# Tier names become `/run-<tier>-ci`; these spellings are taken by other commands.
RESERVED_TIER_NAMES = {"base", "cpu", "failed", "full"}
_TIER_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def gate_labels(workflow: dict) -> set[str]:
    """Labels a workflow requires through its pr-gate.yml calls."""
    labels = set()
    for job in (workflow.get("jobs") or {}).values():
        if job.get("uses") == PR_GATE:
            label = (job.get("with") or {}).get("require-label")
            if label:
                labels.add(label)
    return labels


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
                    f"listed, so /run-{name}-ci never re-runs it"
                )

    if errors:
        print(f"ERROR: {TIERS_FILE} is out of sync with .github/workflows:")
        for error in errors:
            print(f"  {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
