"""The XPU suites /rerun-test runs, read from the PR jobs that run them.

pr-test-xpu.yml runs each suite in a job that starts the CI container with
xpu_ci_start_container.sh, installs the PR with xpu_ci_install_dependency.sh,
and runs `run_suite.py --hw xpu --suite <suite>` in it; /rerun-test runs the
suite's files on that job's runner with the same install.

Run as a script, it prints the install arguments of suite argv[1] as a
GITHUB_OUTPUT line for rerun-test.yml.
"""

import functools
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / ".github/workflows"
WORKFLOWS = ("pr-test-xpu.yml",)
INSTALL_SCRIPT = "scripts/ci/xpu/xpu_ci_install_dependency.sh"

_RUN_SUITE = re.compile(r"\bpython3 run_suite\.py\b(?P<args>.*)", re.S)
_INSTALL = re.compile(re.escape(INSTALL_SCRIPT) + r"(?P<args>[^\n;&|]*)")
_SAFE_ARG = re.compile(r"[\w.=<>\-+]+")


@dataclass(frozen=True)
class XpuSuite:
    runs_on: str
    install_args: tuple  # extra packages the job passes to INSTALL_SCRIPT


def _one(values, what):
    if len(values) != 1:
        raise ValueError(f"{what} has {len(values)} values, not one: {values}")
    return values[0]


@functools.cache
def load_suites():
    """Suite -> XpuSuite for every job in WORKFLOWS that runs a suite after
    INSTALL_SCRIPT; ValueError when the runner or install arguments are not
    literal or one suite runs with two settings."""
    suites = {}
    for name in WORKFLOWS:
        with open(WORKFLOWS_DIR / name, encoding="utf-8") as f:
            jobs = yaml.safe_load(f)["jobs"]
        for job_name, job in jobs.items():
            runs = [str(s.get("run", "")) for s in job.get("steps") or []]
            installs = [m for r in runs for m in _INSTALL.finditer(r)]
            if not installs:
                continue
            source = f"{name} job `{job_name}`"
            install_args = tuple(
                shlex.split(
                    _one(installs, f"{source}'s {INSTALL_SCRIPT} calls")["args"]
                )
            )
            runs_on = job.get("runs-on")
            if (
                not isinstance(runs_on, str)
                or "${{" in runs_on
                or not all(_SAFE_ARG.fullmatch(a) for a in install_args)
            ):
                raise ValueError(f"{source}: non-literal runs-on or install arguments")
            for run in runs:
                m = _RUN_SUITE.search(run.replace("\\\n", " "))
                if m is None or not re.search(r"--hw\s+xpu\b", m.group("args")):
                    continue
                suite = re.search(r"--suite\s+([\w-]+)", m.group("args"))
                if suite is None:
                    raise ValueError(f"{source}: non-literal suite")
                entry = XpuSuite(runs_on, install_args)
                if suites.setdefault(suite.group(1), entry) != entry:
                    raise ValueError(
                        f"XPU suite `{suite.group(1)}` runs as "
                        f"{suites[suite.group(1)]} in one job but as {entry} in "
                        f"{source}"
                    )
    return suites


if __name__ == "__main__":
    suite = load_suites().get(sys.argv[1])
    if suite is None:
        sys.exit(f"No job in {', '.join(WORKFLOWS)} runs XPU suite {sys.argv[1]}")
    print(f"install_args={' '.join(suite.install_args)}")
