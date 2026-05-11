#!/usr/bin/env python3
"""Check that required status check job names are unique across workflows.

Duplicate job names on the same commit allow a passing job in one workflow
to satisfy a required status check meant for a different workflow, bypassing
branch protection.

See: https://github.com/sgl-project/sglang/pull/20208 for an example where
pr-test-npu.yml's "pr-test-finish" job (which passed) caused GitHub to treat
the required "pr-test-finish" check (from pr-test.yml, which failed) as met.
"""

import glob
import sys
from collections import defaultdict

import yaml

# Job names used as required status checks in branch protection.
# These MUST be unique across all workflow files.
PROTECTED_JOB_NAMES = {
    "pr-test-finish",
    "lint",
}


def main() -> int:
    workflows = sorted(glob.glob(".github/workflows/*.yml"))
    job_to_files: dict[str, list[str]] = defaultdict(list)

    for wf in workflows:
        with open(wf, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or "jobs" not in data:
            continue
        for job in data["jobs"]:
            if job in PROTECTED_JOB_NAMES:
                job_to_files[job].append(wf)

    duplicates = {job: files for job, files in job_to_files.items() if len(files) > 1}

    if not duplicates:
        return 0

    print("ERROR: Required status check job names must be unique across workflows.")
    print("Duplicates allow branch protection bypass via auto-merge.\n")
    for job, files in sorted(duplicates.items()):
        print(f"  Job '{job}' appears in:")
        for f in files:
            print(f"    - {f}")
        print()

    print("Fix: rename the job in non-primary workflows to avoid collision.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
