"""Sum est_time per per-commit suite and emit one $GITHUB_OUTPUT line
keyed by suite name. Consumed by pr-test.yml stage jobs as
`fromJson(needs.check-changes.outputs.partitions)['<suite>']`.

    partitions={"stage-b-test-1-gpu-small": {"size": 8, "arr": [0,...,7], "max_parallel": 2}, ...}

Per-shard wall-clock estimate, when a live model is available:

    pred_shard_wall = coeff * sum(est[file] for file in shard) + bias

where `est`, `coeff`, `bias` come from sglang-ci-stats' `model.json`. Each
falls back independently to (a) the in-source `register_X_ci(est_time=N)`
literal for a missing per-file `est`, and (b) `(coeff=1.0, bias=0.0)` for
a missing per-suite fit.
"""

import argparse
import glob
import importlib.util
import json
import math
import os
from collections import defaultdict

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

# Per-partition wall-clock target. ~20 min avg naive; worst-case LPT 4/3
# imbalance is ~27 min, still below the 30-min job-level timeout that acts
# as the real safety net. No LPT slop applied -- we lean on the runtime
# timeout + the explicit MAX_PARTITION_SECONDS sanity check rather than
# padding partition count.
TARGET_SECONDS = 20 * 60

# Hard ceiling. Exceeded -> raise, forcing the maintainer to split a slow file
# or bump TARGET_SECONDS deliberately.
MAX_PARTITION_SECONDS = 30 * 60


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
    """Read sglang-ci-stats' model.json. Returns None when path is missing
    or unparsable -- the caller falls back to (in-source est, coeff=1,
    bias=0) on a per-suite / per-file basis.

    Note the upstream file is called `model.json`; the `partition_model_*`
    naming in sglang disambiguates against ML model checkpoints."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def compute_max_parallel(size: int) -> int:
    return max(size // 4, 1)


def compute_partitions(tests, repo_root, partition_model=None, full_parallel=False):
    """Group per-commit tests by suite and emit partition metadata.

    `partition_model` (optional): parsed `model.json` from sglang-ci-stats.
    Each missing key falls back independently: a file absent from
    `partition_model["est"][suite]` keeps its in-source `est_time`; a
    suite absent from `partition_model["fit"]` uses `(coeff=1.0, bias=0.0)`.

    `full_parallel=True` (scheduled cron or `high priority` PR) sets
    max_parallel = size, lifting the matrix-fanout throttle.
    """
    suite_tests = defaultdict(list)
    for t in tests:
        if t.backend not in _TARGET_BACKENDS:
            continue
        if t.nightly or t.disabled is not None:
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

        fit = fit_table.get(suite)
        coeff = fit["coeff"] if fit else 1.0
        bias = fit["bias"] if fit else 0.0

        # Each shard pays `bias` once (setup / install / warmup overhead),
        # so the per-shard budget for actual test elapsed is
        # (TARGET - bias) / coeff. Solve:
        #     coeff * (total / size) + bias <= TARGET
        # ->  size >= coeff * total / (TARGET - bias)
        if suite in _STAGE_A_OVERRIDES:
            size = _STAGE_A_OVERRIDES[suite]
            max_parallel = size
        else:
            budget = TARGET_SECONDS - bias
            if budget <= 0:
                raise RuntimeError(
                    f"Suite {suite!r}: fit bias={bias}s >= TARGET={TARGET_SECONDS}s. "
                    "Raise TARGET_SECONDS or investigate why this suite's overhead "
                    "alone exceeds the per-shard budget."
                )
            size = max(1, math.ceil(coeff * total / budget))
            max_parallel = size if full_parallel else compute_max_parallel(size)
        # Predicted per-shard wall on naive average (total/size). LPT can
        # be ~4/3 of that in worst case; the 30-min job timeout enforces
        # the real ceiling at runtime. This build-time check fails fast
        # on egregious misconfigs (e.g. stage-a override fixed too small).
        pred_per_shard = coeff * (total / size) + bias
        if pred_per_shard > MAX_PARTITION_SECONDS:
            raise RuntimeError(
                f"Suite {suite!r}: predicted shard wall {pred_per_shard:.0f}s "
                f"(coeff={coeff} * total={total:.0f}/size={size} + bias={bias}) "
                f"exceeds MAX_PARTITION_SECONDS ({MAX_PARTITION_SECONDS}s). "
                f"Split a slow file or raise TARGET_SECONDS deliberately."
            )
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
    args = parser.parse_args()

    files = discover_files(args.repo_root)
    # Warn-not-fail on unregistered files: run_suite.py catches this at
    # test-execution time with sanity_check=True; dispatch should keep going.
    all_tests = collect_tests(files, sanity_check=False)
    partition_model = load_partition_model(args.partition_model_file)

    result = compute_partitions(
        all_tests,
        repo_root=args.repo_root,
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
                src_note = (
                    "live partition model unavailable -- "
                    "all suites fall back to in-source est_time + (coeff=1, bias=0)"
                )
            else:
                src_note = (
                    f"live partition model: "
                    f"`data_as_of={partition_model.get('data_as_of')}`, "
                    f"`n_runs={partition_model.get('n_runs')}`"
                )
            f.write(
                f"`full_parallel={args.full_parallel}` "
                f"(`size//4` throttle is lifted when true); {src_note}\n\n"
            )
            f.write("| Suite | size | max_parallel |\n")
            f.write("|---|---:|---:|\n")
            for suite, info in sorted(result.items()):
                f.write(f"| `{suite}` | {info['size']} | {info['max_parallel']} |\n")


if __name__ == "__main__":
    main()
