"""The AMD PR suites, read from the AMD PR workflows that run them.

pr-test-amd.yml and pr-test-amd-extra.yml hard-code one job per suite. A job
that runs `run_suite.py --hw amd --suite <suite>` through amd_ci_exec.sh and
sets up with the setup-amd-ci action names the suite's runner, ROCm image,
sgl-router fetch, and container env; /rerun-test runs the suite's files the
same way. Every value is the one a pull_request run takes.
"""

import functools
import json
import re
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


def _suite_steps(job):
    """(suite, exec_env) of each step running run_suite.py through amd_ci_exec.sh."""
    out = []
    for step in job.get("steps") or []:
        run = str(step.get("run", "")).replace("\\\n", " ")
        m = _RUN_SUITE.search(run)
        if m is None or not re.search(r"--hw\s+amd\b", m.group("args")):
            continue
        suite = re.search(r"--suite\s+(\S+)", m.group("args"))
        exec_env = tuple(re.findall(r"(?<!\S)-e\s+(\S+)", m.group("exec_args")))
        out.append((suite.group(1) if suite else "", exec_env))
    return out


@functools.cache
def _jobs(workflows_dir=WORKFLOWS_DIR):
    """(source, suite, exec_env, job, jobs) for every job in WORKFLOWS that runs
    one AMD suite."""
    out = []
    for name in WORKFLOWS:
        with open(workflows_dir / name, encoding="utf-8") as f:
            jobs = yaml.safe_load(f)["jobs"]
        for job_name, job in jobs.items():
            runs = set(_suite_steps(job))
            if not runs:
                continue
            source = f"{name} job `{job_name}`"
            suite, exec_env = _one(sorted(runs), f"{source}'s run_suite.py steps")
            if not suite or "${{" in suite:
                raise ValueError(f"{source}: cannot read a literal --suite")
            out.append((source, suite, exec_env, job, jobs))
    return out


@functools.cache
def load_suites(workflows_dir=WORKFLOWS_DIR):
    """Suite -> AmdSuite for every job in WORKFLOWS that sets up with SETUP_ACTION.

    Raises ValueError when a job's values cannot be read, or when two jobs run
    one suite with different settings.
    """
    suites, sources = {}, {}
    for source, suite_name, exec_env, job, jobs in _jobs(workflows_dir):
        setups = [s for s in job["steps"] if s.get("uses") == SETUP_ACTION]
        if not setups:
            continue
        setup = _one(setups, f"{source}'s {SETUP_ACTION} steps").get("with") or {}
        sgl_router = setup.get("sgl_router", False)
        if not isinstance(sgl_router, bool) or any("${{" in e for e in exec_env):
            raise ValueError(
                f"{source}: cannot read a literal sgl_router and container env "
                f"from {setup} and {list(exec_env)}"
            )
        try:
            suite = AmdSuite(
                runs_on=_pr_value(job["runs-on"], job, jobs),
                rocm_version=_pr_value(setup.get("rocm_version", ""), job, jobs),
                sgl_router=sgl_router,
                exec_env=exec_env,
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"{source}: {e!r}") from e
        if suites.setdefault(suite_name, suite) != suite:
            raise ValueError(
                f"AMD suite `{suite_name}` runs with {suites[suite_name]} in "
                f"{sources[suite_name]} but with {suite} in {source}"
            )
        sources.setdefault(suite_name, source)
    return suites


@functools.cache
def own_setup_jobs():
    """Suite -> source for the jobs in WORKFLOWS that set up their own container
    (no SETUP_ACTION step), which /rerun-test cannot reproduce."""
    return {
        suite: source
        for source, suite, _, job, _ in _jobs()
        if not any(s.get("uses") == SETUP_ACTION for s in job["steps"])
    }
