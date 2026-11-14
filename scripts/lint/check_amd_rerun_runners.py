#!/usr/bin/env python3
"""Check AMD_SUITE_RUNNERS in scripts/ci/partition_rerun_tests.py against the AMD PR-test workflows.

`/rerun-test --amd` sends a file to the runner its suite maps to there. Every
suite those workflows run through run_suite.py must map to the runner of the
job that runs it, so a rerun reproduces that job.
"""

import importlib.util
import re
import sys

import yaml

PARTITION_SCRIPT = "scripts/ci/partition_rerun_tests.py"
WORKFLOWS = (
    ".github/workflows/pr-test-amd.yml",
    ".github/workflows/pr-test-amd-extra.yml",
)
# Run by the workflows above but deliberately not dispatchable by /rerun-test.
NOT_RERUNNABLE = {
    # Needs amd_ci_start_container_disagg.sh and the RDMA setup of its own job.
    "stage-b-test-large-8-gpu-mi35x-disaggregation-amd",
}
# `--suite` of test/run_suite.py; the multimodal run_suite.py takes no --hw.
_SUITE_ARG = re.compile(r"run_suite\.py --hw amd --suite ([\w.-]+)")
# runs-on: ${{ format('linux-{0}-1gpu-sglang', inputs.runner_arch || 'mi300') }}
_ARCH_FORMAT = re.compile(
    r"format\('linux-\{0\}-([\w.-]+)', inputs\.runner_arch \|\| '(\w+)'\)"
)


def _runner_labels(job: dict) -> set[str]:
    runs_on = str(job.get("runs-on", ""))
    m = _ARCH_FORMAT.search(runs_on)
    if m:
        return {f"linux-{m.group(2)}-{m.group(1)}"}
    if "matrix.runner" in runs_on:
        return set(job["strategy"]["matrix"]["runner"])
    return {runs_on}


def workflow_suites() -> dict[str, set[str]]:
    """Suite -> runner labels of the jobs that run it."""
    suites: dict[str, set[str]] = {}
    for path in WORKFLOWS:
        with open(path, encoding="utf-8") as f:
            workflow = yaml.safe_load(f)
        for job in workflow["jobs"].values():
            for step in job.get("steps") or []:
                run = " ".join(str(step.get("run", "")).replace("\\\n", " ").split())
                for suite in _SUITE_ARG.findall(run):
                    suites.setdefault(suite, set()).update(_runner_labels(job))
    return suites


def main() -> int:
    spec = importlib.util.spec_from_file_location("partition", PARTITION_SCRIPT)
    partition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(partition)
    runners = partition.AMD_SUITE_RUNNERS
    in_ci = workflow_suites()

    errors = []
    for suite, labels in sorted(in_ci.items()):
        if suite in NOT_RERUNNABLE:
            continue
        if suite not in runners:
            errors.append(
                f"{suite}: run by the AMD PR-test workflows but missing from "
                f"AMD_SUITE_RUNNERS (or add it to NOT_RERUNNABLE here with a reason)"
            )
        elif runners[suite] not in labels:
            errors.append(
                f"{suite}: AMD_SUITE_RUNNERS says `{runners[suite]}` but its job "
                f"runs on {sorted(labels)}"
            )
    for suite in sorted(set(runners) - set(in_ci)):
        errors.append(f"{suite}: in AMD_SUITE_RUNNERS but no AMD PR-test job runs it")

    if errors:
        print(f"ERROR: AMD_SUITE_RUNNERS in {PARTITION_SCRIPT} is out of sync:")
        for error in errors:
            print(f"  {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
