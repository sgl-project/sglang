"""Sum est_time per per-commit suite and emit one $GITHUB_OUTPUT line
keyed by suite name. Consumed by pr-test.yml stage jobs as
`fromJson(needs.check-changes.outputs.partitions)['<suite>']`.

    partitions={"stage-b-test-1-gpu-small": {"size": 8, "arr": [0,...,7], "max_parallel": 2}, ...}
"""

import argparse
import glob
import importlib.util
import json
import math
import os
from collections import defaultdict

import yaml  # PyYAML; preinstalled on ubuntu-latest GHA runners.

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Load ci_register.py directly: `import sglang.test...` pulls torch/orjson via
# sglang.__init__ but check-changes runs on bare ubuntu-latest. ci_register
# itself is stdlib-only (AST).
_CI_REGISTER_PATH = os.path.join(
    REPO_ROOT, "python", "sglang", "test", "ci", "ci_register.py"
)
_spec = importlib.util.spec_from_file_location("ci_register", _CI_REGISTER_PATH)
_ci_register = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ci_register)
collect_tests = _ci_register.collect_tests
HWBackend = _ci_register.HWBackend

# pr-test-amd.yml / pr-test-npu.yml have their own dispatch.
_TARGET_BACKENDS = {HWBackend.CUDA, HWBackend.CPU}

# stage-a is the critical-path entry gate; pin its fanout to smoke-coverage
# defaults instead of est_time. max_parallel = size (no throttle).
_STAGE_A_OVERRIDES = {
    "stage-a-test-cpu": 4,
    "stage-a-test-1-gpu-small": 1,
}

_REUSABLE_STAGE_USES = "./.github/workflows/_pr-test-stage.yml"


def load_run_timeouts(pr_test_yml_path: str) -> dict:
    """Map `self_name -> run_timeout_minutes` from pr-test.yml. The input
    is required in `_pr-test-stage.yml` -- KeyError surfaces missing.
    Inline stage-a-test-cpu is skipped (uses `_STAGE_A_OVERRIDES`)."""
    with open(pr_test_yml_path) as f:
        wf = yaml.safe_load(f)
    timeouts = {}
    for job_id, job in (wf.get("jobs") or {}).items():
        if not isinstance(job, dict) or job.get("uses") != _REUSABLE_STAGE_USES:
            continue
        with_ = job.get("with") or {}
        suite = with_.get("self_name", job_id)
        timeouts[suite] = int(with_["run_timeout_minutes"])
    if not timeouts:
        raise RuntimeError(
            f"load_run_timeouts: no jobs matched uses={_REUSABLE_STAGE_USES!r} "
            f"in {pr_test_yml_path}. The reusable workflow path likely "
            "changed -- update _REUSABLE_STAGE_USES."
        )
    return timeouts


def per_shard_target_seconds(suite: str, run_timeouts: dict) -> float:
    """Per-shard wall budget = 0.75 * stage timeout. 0.75 is the inverse
    of LPT's 4/3 worst-case approximation ratio, so the most imbalanced
    LPT shard fills exactly the timeout."""
    return 0.75 * run_timeouts[suite] * 60


def discover_files(repo_root: str) -> list[str]:
    test_dir = os.path.join(repo_root, "test")
    files = [
        f
        for f in glob.glob(
            os.path.join(test_dir, "registered", "**", "*.py"), recursive=True
        )
        if not f.endswith("/conftest.py") and not f.endswith("/__init__.py")
    ]
    jit_kernel_dir = os.path.join(repo_root, "python", "sglang", "jit_kernel")
    files += glob.glob(
        os.path.join(jit_kernel_dir, "tests", "**", "test_*.py"), recursive=True
    )
    files += glob.glob(
        os.path.join(jit_kernel_dir, "benchmark", "**", "bench_*.py"), recursive=True
    )
    return files


def load_partition_model(path):
    """Read sglang-ci-stats' model.json; None on missing/unparsable.
    Cross-repo schema -- guard against non-dict top-level."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def compute_max_parallel(size: int) -> int:
    return max(size // 3, 1)


def compute_partitions(
    tests, repo_root, run_timeouts, partition_model=None, full_parallel=False
):
    """Group per-commit tests by suite and emit partition metadata.

    `run_timeouts`: `suite -> minutes` from `load_run_timeouts`.
    `partition_model`: optional sglang-ci-stats `model.json`; per-file
    `est` and per-suite `(coeff, bias)` each fall back independently to
    in-source `est_time` / `(1.0, 0.0)`.
    `full_parallel=True` lifts the matrix-fanout throttle.
    """
    # Allowlist: stages pr-test.yml dispatches. Stress / weekly /
    # nightly-* live in test/registered/ but pr-test doesn't run them.
    dispatched_suites = set(run_timeouts) | set(_STAGE_A_OVERRIDES)
    suite_tests = defaultdict(list)
    for t in tests:
        if t.backend not in _TARGET_BACKENDS:
            continue
        if t.nightly or t.disabled is not None:
            continue
        if t.effective_suite not in dispatched_suites:
            continue
        suite_tests[t.effective_suite].append(t)

    est_table = (partition_model or {}).get("est", {})
    fit_table = (partition_model or {}).get("fit", {})

    result = {}
    for suite, group in suite_tests.items():
        live_est = est_table.get(suite, {})
        total = 0.0
        for t in group:
            relpath = os.path.relpath(t.filename, repo_root)
            total += live_est.get(relpath, t.est_time)

        fit = fit_table.get(suite) or {}
        coeff = fit.get("coeff", 1.0)
        bias = fit.get("bias", 0.0)

        # Each shard pays `bias` once, so size >= coeff*total / (target-bias).
        if suite in _STAGE_A_OVERRIDES:
            size = _STAGE_A_OVERRIDES[suite]
            max_parallel = size
        else:
            target = per_shard_target_seconds(suite, run_timeouts)
            budget = target - bias
            if budget <= 0:
                raise RuntimeError(
                    f"Suite {suite!r}: fit bias={bias}s >= target={target}s. "
                    "Investigate the fit or raise the stage's run_timeout_minutes."
                )
            ideal_size = math.ceil(coeff * total / budget)
            # ideal_size > len(group) -> slowest single file alone exceeds
            # the per-shard budget; surface via raise instead of empty shard.
            if ideal_size > len(group):
                raise RuntimeError(
                    f"Suite {suite!r}: needs {ideal_size} shards but has only "
                    f"{len(group)} test file(s). target={target:.0f}s, "
                    f"coeff={coeff}, bias={bias}s, total_est={total:.0f}s."
                )
            size = max(1, ideal_size)
            max_parallel = size if full_parallel else compute_max_parallel(size)
        result[suite] = {
            "size": size,
            "arr": list(range(size)),
            "max_parallel": max_parallel,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=REPO_ROOT)
    parser.add_argument(
        "--output-format",
        choices=("gha", "json"),
        default="gha",
        help="`gha` emits `partitions=<json>` for $GITHUB_OUTPUT; `json` is raw",
    )
    parser.add_argument(
        "--full-parallel",
        choices=("true", "false"),
        default="false",
        help="Lift the max_parallel throttle (set by schedule / `high priority`)",
    )
    parser.add_argument(
        "--partition-model-file",
        default=None,
        help="Path to sglang-ci-stats model.json (omit/missing -> static fallback)",
    )
    parser.add_argument(
        "--pr-test-yml",
        default=os.path.join(REPO_ROOT, ".github", "workflows", "pr-test.yml"),
        help="Path to pr-test.yml; per-stage `run_timeout_minutes` is read from here.",
    )
    args = parser.parse_args()

    files = discover_files(args.repo_root)
    # Warn-not-fail on unregistered files: run_suite.py catches this at
    # test-execution time with sanity_check=True; dispatch should keep going.
    all_tests = collect_tests(files, sanity_check=False)
    partition_model = load_partition_model(args.partition_model_file)
    run_timeouts = load_run_timeouts(args.pr_test_yml)

    result = compute_partitions(
        all_tests,
        repo_root=args.repo_root,
        run_timeouts=run_timeouts,
        partition_model=partition_model,
        full_parallel=(args.full_parallel == "true"),
    )
    payload = json.dumps(result, separators=(",", ":"), sort_keys=True)
    if args.output_format == "gha":
        print(f"partitions={payload}")
    else:
        print(payload)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("## Partitions\n\n")
            if partition_model is None:
                src_note = "no live model -- static est_time + (coeff=1, bias=0)"
            else:
                src_note = (
                    f"live model `data_as_of={partition_model.get('data_as_of')}`, "
                    f"`n_runs={partition_model.get('n_runs')}`"
                )
            f.write(
                f"`full_parallel={args.full_parallel}` "
                f"(`size//3` throttle is lifted when true); {src_note}\n\n"
            )
            f.write("| Suite | size | max_parallel |\n")
            f.write("|---|---:|---:|\n")
            for suite, info in sorted(result.items()):
                f.write(f"| `{suite}` | {info['size']} | {info['max_parallel']} |\n")


if __name__ == "__main__":
    main()
