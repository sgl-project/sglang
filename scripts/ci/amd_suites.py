"""The AMD suites /rerun-test runs, read from the PR jobs that run them.

pr-test-amd.yml and pr-test-amd-extra.yml run each suite in a job that sets up
with the setup-amd-ci action and runs `run_suite.py --hw amd --suite <suite>`
through amd_ci_exec.sh; /rerun-test runs the suite's files the same way, with
the values a pull_request run takes. Jobs that set up their own container
(disaggregation) are left out.

Run as a script, it prints the setup inputs and container env of suite argv[1]
as GITHUB_OUTPUT lines for rerun-test.yml.
"""

import functools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / ".github/workflows"
WORKFLOWS = ("pr-test-amd.yml", "pr-test-amd-extra.yml")
SETUP_ACTION = "./.github/actions/setup-amd-ci"

_EXPR = re.compile(r"\$\{\{\s*(.*?)\s*\}\}", re.S)
_RUN_SUITE = re.compile(
    r"amd_ci_exec\.sh(?P<exec_args>.*?)\bpython3 run_suite\.py\b(?P<args>.*)", re.S
)


@dataclass(frozen=True)
class AmdSuite:
    runs_on: str
    rocm_version: str
    sgl_router: bool
    exec_env: tuple  # KEY=VALUE pairs the job passes to amd_ci_exec.sh with -e


def _one(values, what):
    if len(values) != 1:
        raise ValueError(f"{what} has {len(values)} values, not one: {values}")
    return values[0]


def _pr_value(value, job, jobs):
    """The value `value` takes in a pull_request run of `job`.

    A pull_request run has no `inputs`, so every `<cond> && '<x>' || '<y>'`
    chain these workflows use falls through to its last literal. A one-value
    list (a matrix axis) stands for its value.
    """
    if isinstance(value, list):
        return _pr_value(_one(value, "list"), job, jobs)
    value = str(value)
    m = _EXPR.fullmatch(value.strip())
    if m is None:
        if "${{" in value:
            raise ValueError(f"cannot evaluate `{value}`")
        return value
    return _pr_expr(m.group(1), job, jobs)


def _pr_expr(expr, job, jobs):
    if m := re.fullmatch(r"format\('([^']*)',\s*(.+)\)", expr, re.S):
        return m.group(1).replace("{0}", _pr_expr(m.group(2), job, jobs))
    if m := re.fullmatch(r"fromJson\((.+)\)", expr, re.S):
        return _one(json.loads(_pr_expr(m.group(1), job, jobs)), expr)
    if m := re.fullmatch(r"matrix\.([\w-]+)", expr):
        return _pr_value(job["strategy"]["matrix"][m.group(1)], job, jobs)
    if m := re.fullmatch(r"needs\.([\w-]+)\.outputs\.([\w-]+)", expr):
        return _pr_value(jobs[m.group(1)]["outputs"][m.group(2)], job, jobs)
    if m := re.search(r"\|\|\s*'([^']*)'$", expr):
        return m.group(1)
    raise ValueError(f"cannot evaluate `${{{{ {expr} }}}}`")


@functools.cache
def load_suites():
    """Suite -> AmdSuite for every job in WORKFLOWS that sets up with
    SETUP_ACTION; ValueError when a value is not literal for a pull_request run
    or one suite runs with two settings."""
    suites = {}
    for name in WORKFLOWS:
        with open(WORKFLOWS_DIR / name, encoding="utf-8") as f:
            jobs = yaml.safe_load(f)["jobs"]
        for job_name, job in jobs.items():
            steps = job.get("steps") or []
            setups = [s for s in steps if s.get("uses") == SETUP_ACTION]
            if not setups:
                continue
            source = f"{name} job `{job_name}`"
            setup = _one(setups, f"{source}'s {SETUP_ACTION} steps").get("with") or {}
            for step in steps:
                run = str(step.get("run", "")).replace("\\\n", " ")
                m = _RUN_SUITE.search(run)
                if m is None or not re.search(r"--hw\s+amd\b", m.group("args")):
                    continue
                suite = re.search(r"--suite\s+(\S+)", m.group("args"))
                exec_env = tuple(re.findall(r"(?<!\S)-e\s+(\S+)", m.group("exec_args")))
                sgl_router = setup.get("sgl_router", False)
                if (
                    suite is None
                    or not isinstance(sgl_router, bool)
                    or "${{" in suite.group(1) + " ".join(exec_env)
                ):
                    raise ValueError(
                        f"{source}: non-literal suite, sgl_router, or container env"
                    )
                try:
                    entry = AmdSuite(
                        _pr_value(job["runs-on"], job, jobs),
                        _pr_value(setup.get("rocm_version", ""), job, jobs),
                        sgl_router,
                        exec_env,
                    )
                except KeyError as e:
                    raise ValueError(f"{source}: cannot evaluate {e!r}") from e
                if suites.setdefault(suite.group(1), entry) != entry:
                    raise ValueError(
                        f"AMD suite `{suite.group(1)}` runs as "
                        f"{suites[suite.group(1)]} in one job but as {entry} in "
                        f"{source}"
                    )
    return suites


if __name__ == "__main__":
    suite = load_suites().get(sys.argv[1])
    if suite is None:
        sys.exit(f"No {SETUP_ACTION} job in {', '.join(WORKFLOWS)} runs {sys.argv[1]}")
    print(f"rocm_version={suite.rocm_version}")
    print(f"sgl_router={str(suite.sgl_router).lower()}")
    print(f"exec_env={' '.join(suite.exec_env)}")
