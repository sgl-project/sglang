"""The NPU suites /rerun-test runs, read from the CI jobs that run them.

pr-test-npu.yml, nightly-test-npu.yml, and full-test-npu.yml call
_npu-pr-test-stage.yml once per suite; /rerun-test calls it the same way, with
the PR installed into the a3 image of pr-test-npu.yml.

Run as a script, it prints the stage inputs of suite argv[1] as GITHUB_OUTPUT
lines for rerun-test.yml.
"""

import functools
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / ".github/workflows"
WORKFLOWS = ("pr-test-npu.yml", "nightly-test-npu.yml", "full-test-npu.yml")
STAGE = "./.github/workflows/_npu-pr-test-stage.yml"
# pr-test-npu.yml switches runners for coverage mode:
# `${{ inputs.coverage_mode == true && '<coverage>' || '<normal>' }}`.
_NORMAL_RUNNER = re.compile(r"^\$\{\{.*\|\|\s*'([^']+)'\s*}}$")


@dataclass(frozen=True)
class NpuSuite:
    runs_on: str
    sgl_router: bool
    nightly: bool  # its tests are registered with nightly=True


@functools.cache
def load_suites():
    """Suite -> NpuSuite for every a3 job in WORKFLOWS that calls STAGE."""
    suites = {}
    for name in WORKFLOWS:
        with open(WORKFLOWS_DIR / name, encoding="utf-8") as f:
            jobs = yaml.safe_load(f)["jobs"]
        for job_name, job in jobs.items():
            inputs = job.get("with") or {}
            if job.get("uses") != STAGE or inputs.get("npu_device_type", "a3") != "a3":
                continue
            suite = str(inputs["self_name"])
            runner = str(inputs["runner_config"])
            m = _NORMAL_RUNNER.match(runner)
            runs_on = m.group(1) if m else runner
            if "${{" in suite + runs_on:
                raise ValueError(
                    f"{name} job `{job_name}`: non-literal suite or runner"
                )
            entry = NpuSuite(
                runs_on,
                inputs.get("sgl_router", False),
                inputs.get("is_nightly_pipeline_job", False),
            )
            if suites.setdefault(suite, entry) != entry:
                raise ValueError(
                    f"NPU suite `{suite}` runs as {suites[suite]} in one job but "
                    f"as {entry} in {name} job `{job_name}`"
                )
    return suites


def a3_pr_image():
    """The CANN image pr-test-npu.yml installs the PR into on a3 runners."""
    with open(WORKFLOWS_DIR / "pr-test-npu.yml", encoding="utf-8") as f:
        return yaml.safe_load(f)["env"]["NPU_A3_PR_IMAGE"]


if __name__ == "__main__":
    suite = load_suites().get(sys.argv[1])
    if suite is None:
        sys.exit(f"No a3 {STAGE} job in {', '.join(WORKFLOWS)} runs {sys.argv[1]}")
    print(f"image={a3_pr_image()}")
    print(f"sgl_router={str(suite.sgl_router).lower()}")
    print(f"collect_nightly_tests={str(suite.nightly).lower()}")
