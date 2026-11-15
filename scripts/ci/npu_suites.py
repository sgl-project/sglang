"""The single-node NPU suites, read from the NPU workflows that run them.

pr-test-npu.yml, nightly-test-npu.yml, and full-test-npu.yml hard-code one job
per suite. Each job that calls an NPU stage workflow names its suite, runner,
and whether it builds sgl-router; /rerun-test runs a suite the same way.
"""

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / ".github/workflows"
WORKFLOWS = ("pr-test-npu.yml", "nightly-test-npu.yml", "full-test-npu.yml")
PR_WORKFLOW = "pr-test-npu.yml"
# Workflow-level env of pr-test-npu.yml: the CANN image PR runs install into.
A3_PR_IMAGE_ENV = "NPU_A3_PR_IMAGE"
# Stage workflow -> (suite input, runner input).
STAGES = {
    "_npu-pr-test-stage.yml": ("self_name", "runner_config"),
    "_npu-single-node-test-stage.yml": ("test_suite", "runner"),
}
# pr-test-npu.yml switches runners for coverage mode:
# `${{ inputs.coverage_mode == true && '<coverage>' || '<normal>' }}`.
_NORMAL_RUNNER = re.compile(r"\|\|\s*'([^']+)'\s*}}$")


@dataclass(frozen=True)
class NpuSuite:
    stage: str  # stage workflow file the suite's job calls
    runs_on: str
    sgl_router: bool


def _runner_label(value):
    m = _NORMAL_RUNNER.search(value)
    return m.group(1) if m and value.startswith("${{") else value


def _read(workflows_dir, name):
    with open(workflows_dir / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.cache
def load_suites(workflows_dir=WORKFLOWS_DIR):
    """Suite -> NpuSuite for every job in WORKFLOWS that calls a stage workflow.

    Raises ValueError when a job's values cannot be read, or when two jobs run
    one suite with different settings.
    """
    suites, sources = {}, {}
    for name in WORKFLOWS:
        for job_name, job in _read(workflows_dir, name)["jobs"].items():
            stage = (job.get("uses") or "").removeprefix("./.github/workflows/")
            if stage not in STAGES:
                continue
            source = f"{name} job `{job_name}`"
            suite_key, runner_key = STAGES[stage]
            inputs = job.get("with") or {}
            suite_name = str(inputs.get(suite_key, ""))
            runs_on = _runner_label(str(inputs.get(runner_key, "")))
            sgl_router = inputs.get("sgl_router", False)
            if (
                not suite_name
                or "${{" in suite_name
                or not runs_on
                or "${{" in runs_on
                or not isinstance(sgl_router, bool)
            ):
                raise ValueError(
                    f"{source}: cannot read a literal suite (`{suite_key}`), runner "
                    f"(`{runner_key}`), and sgl_router from {inputs}"
                )
            suite = NpuSuite(stage, runs_on, sgl_router)
            if suites.setdefault(suite_name, suite) != suite:
                raise ValueError(
                    f"NPU suite `{suite_name}` runs with {suites[suite_name]} in "
                    f"{sources[suite_name]} but with {suite} in {source}"
                )
            sources.setdefault(suite_name, source)
    return suites


@functools.cache
def a3_pr_image():
    """The CANN image pr-test-npu.yml installs the PR into on a3 runners."""
    env = _read(WORKFLOWS_DIR, PR_WORKFLOW).get("env") or {}
    if A3_PR_IMAGE_ENV not in env:
        raise ValueError(f"{PR_WORKFLOW} has no workflow-level env {A3_PR_IMAGE_ENV}")
    return env[A3_PR_IMAGE_ENV]
