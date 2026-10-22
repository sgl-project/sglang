import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import tabulate

from sglang.test.ci.ci_register import (
    CIRegistry,
    HWBackend,
    auto_partition,
    collect_tests,
)
from sglang.test.ci.ci_utils import run_unittest_files

HW_MAPPING = {
    "cpu": HWBackend.CPU,
    "cuda": HWBackend.CUDA,
    "amd": HWBackend.AMD,
    "musa": HWBackend.MUSA,
    "npu": HWBackend.NPU,
    "xpu": HWBackend.XPU,
    "mlx": HWBackend.MLX,
    "ppu": HWBackend.PPU,
}

# Per-commit test suites (run on every PR).
# Includes both base-a/b/c (always-on; pr-test.yml) and extra-a/b
# (label-gated; pr-test-extra.yml). Tests are tagged per-commit regardless;
# pr-test-extra.yml's `run-ci-extra` PR label decides whether extra-* dispatches.
PER_COMMIT_SUITES = {
    HWBackend.CPU: [
        "base-a-test-cpu",
        "base-b-test-cpu",
        "base-c-test-cpu",
        "base-b-test-cpu-arm64",
        "stage-a-test-cpu-ppu",
    ],
    HWBackend.AMD: [
        "stage-a-test-1-gpu-small-amd",
        "stage-b-test-1-gpu-small-amd",
        "stage-b-test-1-gpu-small-amd-nondeterministic",
        "stage-b-test-1-gpu-small-amd-mi35x",
        "stage-b-test-large-8-gpu-mi35x-disaggregation-amd",
        "stage-b-test-1-gpu-large-amd",
        "stage-b-test-2-gpu-large-amd",
        "jit-kernel-unit-test-amd",
        "jit-kernel-benchmark-test-amd",
        "sgl-kernel-unit-test-2-gpu-amd",
        "stage-c-test-4-gpu-amd",
        "stage-c-test-large-8-gpu-amd",
        "stage-c-test-large-8-gpu-amd-mi35x",
        # extra-a: label-gated PR opt-in suites in pr-test-amd-extra.yml
        # (mirror of CUDA extra-a; tests stay tagged per-commit but only
        # dispatch when the PR carries the `run-ci-extra` label). 1-gpu-small
        # carries the mock-model / kv_canary unit + single-GPU canary e2e
        # tests; 1-gpu-large carries the subset of model e2e tests validated
        # to pass on mi325 (quant fp8kv-triton, sessions streaming-session
        # EAGLE3, spec standalone triton-backend variant); 2-gpu-large carries
        # the multi-GPU (TP/PP/PD) mock-model + kv_canary e2e tests. The rest
        # of CUDA extra-a tests fail on ROCm (missing flash_attn.cute/flash_ops
        # kernels, OOM, or accuracy regressions — e.g. gemma4-mtp-31b dips
        # below the gsm8k floor on the topk=3 leg) and stay CUDA-only for now.
        "extra-a-test-1-gpu-small-amd",
        "extra-a-test-1-gpu-large-amd",
        "extra-a-test-2-gpu-large-amd",
    ],
    HWBackend.MUSA: [],
    HWBackend.CUDA: [
        "base-a-test-1-gpu-small",
        "base-b-test-1-gpu-small",
        "base-b-test-1-gpu-large",
        "base-b-test-2-gpu-large",
        "base-b-test-4-gpu-b200",
        "base-b-kernel-unit-test-1-gpu-large",
        "base-b-kernel-unit-test-1-gpu-b200",
        "base-b-kernel-unit-test-4-gpu-b200",
        "base-b-kernel-unit-test-8-gpu-h200",
        "base-b-kernel-benchmark-test-1-gpu-large",
        "base-c-test-4-gpu-h100",
        "base-c-test-4-gpu-b200",
        "base-c-test-4-gpu-gb300",
        "base-c-test-8-gpu-h20",
        "base-c-test-8-gpu-h200",
        "base-c-test-8-gpu-b200",
        "base-c-test-8-gpu-b300",
        # extra-a / extra-b: label-gated PR opt-in suites in pr-test-extra.yml
        # (tests still tagged per-commit but skipped on default PR runs).
        "extra-a-test-1-gpu-small",
        "extra-a-test-1-gpu-large",
        "extra-a-test-2-gpu-large",
        "extra-b-test-4-gpu-h100",
        "extra-b-test-4-gpu-b200",
        "extra-b-test-8-gpu-h200",
    ],
    HWBackend.NPU: [
        "base-a-test-1-npu-a2",
        "base-b-test-1-npu-a3",
        "base-b-test-2-npu-a3",
        "base-b-test-4-npu-a3",
        "base-b-test-8-npu-a3",
        "base-b-test-16-npu-a3",
        "base-c-test-acc-2-npu-a3",
        "base-c-test-acc-4-npu-a3",
        "base-c-test-acc-8-npu-a3",
        "base-c-test-acc-16-npu-a3",
        "base-c-test-perf-2-npu-a3",
        "base-c-test-perf-4-npu-a3",
        "base-c-test-perf-8-npu-a3",
        "base-c-test-perf-16-npu-a3",
    ],
    HWBackend.XPU: [
        "stage-a-test-1-gpu-xpu",
        "stage-b-test-1-gpu-xpu",
    ],
    HWBackend.MLX: [
        "stage-a-unit-test-mlx",
        "stage-b-e2e-mlx",
    ],
    HWBackend.PPU: [
        # CPU/smoke precede parallel bodies; stage-c keeps its legacy name.
        "stage-a-test-1-gpu-ppu",
        "stage-b-test-1-gpu-ppu",
        "stage-b-kernel-test-1-gpu-ppu",
        "stage-c-test-2-gpu-ppu",
    ],
}

# Nightly test suites (run nightly, organized by GPU configuration)
NIGHTLY_SUITES = {
    HWBackend.CUDA: [
        # `stage="nightly"` + a runner_config, same `{stage}-test-{runner_config}`
        # shape as the per-commit suites. No `nightly=True`: the stage name
        # carries the cadence; only the legacy suites below still need the flag.
        "nightly-test-1-gpu-large",
        "nightly-test-2-gpu-large",
        "nightly-test-4-gpu-h100",
        "nightly-test-4-gpu-b200",
        "nightly-test-4-gpu-gb300",
        "nightly-test-8-gpu-h200",
        "nightly-test-8-gpu-b200",
    ],
    HWBackend.AMD: [
        "nightly-amd",
        "nightly-amd-1-gpu",
        "nightly-amd-kernel-1-gpu",
        "nightly-amd-1-gpu-mi35x",
        "nightly-amd-1-gpu-zimage-turbo",
        "nightly-amd-2-gpu-mi35x-deepseek-r1-mxfp4-tp2",
        "nightly-amd-8-gpu-mi35x-deepseek-r1-mxfp4-tp4",
        "nightly-amd-accuracy-8-gpu-mi35x-kimi-k3",
        "nightly-amd-4-gpu",
        "nightly-amd-8-gpu",
        "nightly-amd-vlm",
        "nightly-amd-8-gpu-mi35x-deepseek-v4-flash",
        # MI35x 8-GPU suite (different model configs)
        "nightly-amd-8-gpu-mi35x",
    ],
    HWBackend.MUSA: [
        "nightly-musa-1-gpu",
    ],
    HWBackend.CPU: [],
    HWBackend.NPU: [
        "nightly-1-npu-a3",
        "nightly-2-npu-a3",
        "nightly-4-npu-a3",
        "nightly-8-npu-a3",
        "nightly-16-npu-a3",
        "full-1-npu-a3",
        "full-2-npu-a3",
        "full-4-npu-a3",
        "full-8-npu-a3",
        "full-16-npu-a3",
    ],
    HWBackend.XPU: [
        "nightly-xpu-1-gpu",
        "nightly-xpu-2-gpu",
        "nightly-xpu-4-gpu",
    ],
    HWBackend.PPU: [
        "nightly-1-ppu",
        "nightly-2-ppu",
        "nightly-4-ppu",
        "nightly-8-ppu",
        # Answer-quality suites, kept out of the nightly-1..8 serial chain and
        # driven by their own workflows -- the test-ppu-answer* K8s files that
        # nightly-test-ppu-answer.yml orchestrates -- so a multi-hour accuracy
        # run holds its own K8s boards instead of the nightly-1..8 `ppu` runner.
        # Listed here so the suite names are registered ones and
        # run_suite.py / the coverage report account for their tests.
        "nightly-answer-1-ppu",
        "nightly-answer-8-ppu",
        # One suite per model, not one per entry: a suite is what --suite runs,
        # and every file in it executes in the same process off the same
        # SGLANG_PPU_ANSWER_TEST_CONFIG, so two models could not be given
        # different checkpoints under one name. Several configs of the same model
        # do share a suite -- the workflow picks which one by naming its config.
        # Qwen3.5's four configs are why nightly-answer-8-ppu is not split.
        "nightly-answer-8-glm52-ppu",
        "nightly-answer-8-kimi26-ppu",
        "nightly-answer-8-minimax27-ppu",
        # Whole boards, one suite each: every node runs the same registered file
        # and the launcher tells each which rank it is.
        "nightly-answer-16-ppu",
        "nightly-answer-16-kimi26-ppu",
        "nightly-answer-32-ppu",
        # Serving performance suites, driven by their own workflows for the same
        # reason the answer ones are: a measurement holds a whole board for as
        # long as the checkpoint takes to load, and the nightly-1..8 chain must
        # not wait behind that. A suite records TTFT and throughput and judges
        # neither -- it is red only when a measurement produced no numbers -- so
        # the names below carry evidence, not a threshold.
        #
        # The number is the devices the configs of that suite declare, not the
        # board they hold: the plan serves MiniMax-M2.7 at tp 2 and Qwen3.5 and
        # GLM-5.2-MXFP4 at tp 4, and one suite cannot state two device counts,
        # which is why GLM-5.2 has two suites while its answer line has one.
        "nightly-perf-2-minimax27-ppu",
        "nightly-perf-4-glm52-ppu",
        "nightly-perf-4-qwen35-ppu",
        "nightly-perf-8-glm52-ppu",
        "nightly-perf-8-kimi26-ppu",
        # Whole boards, one suite each.
        "nightly-perf-16-ppu",
        "nightly-perf-32-ppu",
        # Prefill/decode-disaggregated serving, which is two servers of one
        # checkpoint plus a router and so always holds more than one board: the
        # number is again the devices the config declares, sixteen for a 1p1d of
        # two eight-device boards. Its own suite rather than an entry in
        # nightly-perf-16-ppu because the two lines run different code -- a PD
        # node's role, and whether it measures at all, depends on the rank the
        # launcher handed it -- and its own workflow for the reason every perf
        # suite has one.
        "nightly-pd-perf-16-glm52-ppu",
        # The same PD story for the second checkpoint under review; its own suite
        # because it is a different model and config, not another line of glm52's.
        "nightly-pd-perf-16-qwen35-ppu",
        # Public-benchmark accuracy suites, scored by EvalScope against a served
        # model and driven by test-ppu-accuracy.yml. Kept out of the
        # nightly-1..8 chain for the same reason as the two lines above, and more
        # so: a full split is 1319 prompts against a reasoning model, which is
        # most of a night on one board.
        #
        # Unlike the perf line these do judge, when a baseline exists: the config
        # states a ratio band around a recorded score and a run outside it is red.
        # Where no baseline has been recorded on this hardware yet the score is
        # published and nothing is judged, which the report says in as many words.
        #
        # One per model for the reason the answer comment above gives. Each holds
        # that model's quantisations across all three datasets -- eight configs
        # behind three of these names, four behind Kimi's -- and the workflow names
        # which one runs. The number is 8 throughout because every config declares
        # eight devices, including the two models the internal plan serves at tp 2
        # and tp 4: a full split is generation-bound and the job holds the whole
        # board either way.
        "nightly-accuracy-8-glm52-ppu",
        "nightly-accuracy-8-kimi26-ppu",
        "nightly-accuracy-8-minimax27-ppu",
        "nightly-accuracy-8-qwen35-ppu",
    ],
}


OTHER_SUITES = {
    HWBackend.CPU: [
        "default",
    ],
    HWBackend.CUDA: [
        "stress",
        # `stage="weekly"` -- same shape. The three dicts group names for
        # readability only; validation reads their union.
        "weekly-test-8-gpu-h200",
    ],
}


_SUITE_CHECKED_BACKENDS = {
    HWBackend.CUDA,
    HWBackend.CPU,
    HWBackend.MUSA,
    HWBackend.XPU,
    HWBackend.MLX,
    HWBackend.PPU,
}


def _valid_suites_by_backend() -> dict:
    """Build a mapping from backend to its set of valid suite names."""
    result = {}
    for suite_dict in (PER_COMMIT_SUITES, NIGHTLY_SUITES, OTHER_SUITES):
        for backend, suites in suite_dict.items():
            if backend not in result:
                result[backend] = set()
            result[backend].update(suites)
    return result


def validate_all_suites(all_tests: List[CIRegistry]):
    """Fail fast if any test is registered to a suite that doesn't belong to its backend."""
    valid_by_backend = _valid_suites_by_backend()
    errors = []
    for t in all_tests:
        if t.backend not in _SUITE_CHECKED_BACKENDS:
            continue
        valid = valid_by_backend.get(t.backend, set())
        if t.effective_suite not in valid:
            errors.append(
                f"  {t.filename}: backend={t.backend.name}, suite='{t.effective_suite}'"
            )
    if errors:
        raise ValueError("Tests registered to invalid suites:\n" + "\n".join(errors))


def filter_tests(
    ci_tests: List[CIRegistry], hw: HWBackend, suite: str, nightly: bool = False
) -> List[CIRegistry]:
    ci_tests = [
        t
        for t in ci_tests
        if t.backend == hw and t.effective_suite == suite and t.nightly == nightly
    ]

    # Union of all three dicts, not just the per-commit or nightly half:
    # CUDA nightly suites are selected by name alone, without --nightly.
    if suite not in _valid_suites_by_backend().get(hw, set()):
        print(f"Warning: Unknown suite {suite} for backend {hw.name}")

    enabled_tests = [t for t in ci_tests if t.disabled is None]
    skipped_tests = [t for t in ci_tests if t.disabled is not None]

    return enabled_tests, skipped_tests


def pretty_print_tests(
    args, ci_tests: List[CIRegistry], skipped_tests: List[CIRegistry]
):
    hw = HW_MAPPING[args.hw]
    suite = args.suite
    nightly = args.nightly
    if args.auto_partition_size:
        partition_info = (
            f"{args.auto_partition_id + 1}/{args.auto_partition_size} "
            f"(0-based id={args.auto_partition_id})"
        )
    else:
        partition_info = "full"

    headers = ["Hardware", "Suite", "Nightly", "Partition"]
    rows = [[hw.name, suite, str(nightly), partition_info]]
    msg = tabulate.tabulate(rows, headers=headers, tablefmt="psql") + "\n"

    if skipped_tests:
        msg += f"⚠️  Skipped {len(skipped_tests)} test(s):\n"
        for t in skipped_tests:
            reason = t.disabled or "disabled"
            msg += f"  - {t.filename} (reason: {reason})\n"
        msg += "\n"

    if len(ci_tests) == 0:
        msg += f"No tests found for hw={hw.name}, suite={suite}, nightly={nightly}\n"
        msg += "This is expected during incremental migration. Skipping.\n"
    else:
        total_est_time = sum(t.est_time for t in ci_tests)
        msg += (
            f"✅ Enabled {len(ci_tests)} test(s) (est total {total_est_time:.1f}s):\n"
        )
        for t in ci_tests:
            msg += f"  - {t.filename} (est_time={t.est_time})\n"

    print(msg, flush=True)


def load_live_est(
    partition_model_file: Optional[str], suite: str, repo_root: str
) -> Optional[Dict[str, float]]:
    """`CIRegistry.filename -> est seconds` from `model.json est[suite]`;
    None on any miss (caller falls back to in-source `est_time`)."""
    if not partition_model_file or not os.path.exists(partition_model_file):
        return None
    try:
        with open(partition_model_file) as f:
            partition_model = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(partition_model, dict):
        return None
    suite_est = partition_model.get("est", {}).get(suite)
    if not isinstance(suite_est, dict) or not suite_est:
        return None
    return {
        os.path.join(repo_root, relpath): float(elapsed)
        for relpath, elapsed in suite_est.items()
    }


def run_a_suite(args):
    hw = HW_MAPPING[args.hw]
    suite = args.suite
    nightly = args.nightly
    auto_partition_id = args.auto_partition_id
    auto_partition_size = args.auto_partition_size

    # Use absolute paths so the script works from any working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    # Registered tests under test/registered/
    files = [
        f
        for f in glob.glob(
            os.path.join(script_dir, "registered", "**", "*.py"), recursive=True
        )
        # conftest.py / __init__.py are pytest+package structure, never
        # registered tests, and must not be executed as one.
        if os.path.basename(f) not in ("conftest.py", "__init__.py")
    ]

    # Strict: all discovered files must have proper registration
    sanity_check = True

    all_tests = collect_tests(files, sanity_check=sanity_check)
    validate_all_suites(all_tests)
    ci_tests, skipped_tests = filter_tests(all_tests, hw, suite, nightly)
    if getattr(args, "require_nonempty", False) and skipped_tests:
        raise ValueError("Strict PR selection contains disabled registrations")

    if auto_partition_size:
        live_est = load_live_est(args.partition_model_file, suite, repo_root)
        if live_est is not None:
            print(
                f"LPT: {len(live_est)} live est entries from {args.partition_model_file}",
                flush=True,
            )
        else:
            print(
                f"LPT: no live est ({args.partition_model_file!r}); using in-source est_time",
                flush=True,
            )
        ci_tests = auto_partition(
            ci_tests, auto_partition_id, auto_partition_size, live_est=live_est
        )

    pretty_print_tests(args, ci_tests, skipped_tests)

    # None hands the per-file budget over to est_time (see run_unittest_files).
    timeout = None if args.timeout_from_est_time else args.timeout_per_file

    # Add extra timeout when retry is enabled
    if timeout is not None and args.enable_retry:
        timeout += args.retry_timeout_increase

    return run_unittest_files(
        ci_tests,
        timeout_per_file=timeout,
        continue_on_error=args.continue_on_error,
        enable_retry=args.enable_retry,
        max_attempts=args.max_attempts,
        retry_wait_seconds=args.retry_wait_seconds,
        **(
            {
                "pytest_mode": args.pytest,
                "require_nonempty": args.require_nonempty,
                "junit_dir": args.junit_dir,
                "suite_timeout": args.suite_timeout,
                "allowed_skips": tuple(args.allow_skip),
            }
            if getattr(args, "pytest", False)
            or getattr(args, "require_nonempty", False)
            else {}
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run CI test suites from test/registered/"
    )
    parser.add_argument(
        "--hw",
        type=str,
        choices=HW_MAPPING.keys(),
        required=True,
        help="Hardware backend to run tests on.",
    )
    parser.add_argument("--suite", type=str, required=True, help="Test suite to run.")
    parser.add_argument(
        "--nightly",
        action="store_true",
        help=(
            "Include tests registered with nightly=True (AMD/CPU/NPU). CUDA "
            "scheduled suites are selected by name and take no flag."
        ),
    )
    parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1200,
        help="The time limit for running one file in seconds (default: 1200).",
    )
    parser.add_argument(
        "--timeout-from-est-time",
        action="store_true",
        help=(
            "Derive each file's time limit from its own est_time instead of "
            "the flat --timeout-per-file, for suites mixing fast and slow tests."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining tests even if one fails (default: False, useful for nightly tests).",
    )
    parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    parser.add_argument(
        "--enable-retry",
        action="store_true",
        default=False,
        help="Enable smart retry for accuracy/performance assertion failures (not code errors)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum number of attempts per file including initial run (default: 2)",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=int,
        default=60,
        help="Seconds to wait between retries (default: 60)",
    )
    parser.add_argument(
        "--retry-timeout-increase",
        type=int,
        default=600,
        help="Additional timeout in seconds when retry is enabled (default: 600)",
    )
    parser.add_argument(
        "--partition-model-file",
        type=str,
        default=None,
        help="Path to sglang-ci-stats model.json for live LPT est; missing/malformed -> in-source est_time fallback.",
    )
    parser.add_argument(
        "--pytest",
        action="store_true",
        help="Strict per-file pytest/JUnit execution (no retries).",
    )
    parser.add_argument("--require-nonempty", action="store_true")
    parser.add_argument("--junit-dir")
    parser.add_argument("--suite-timeout", type=float)
    parser.add_argument(
        "--allow-skip",
        action="append",
        default=[],
        help="Exact JUnit classname.method of a documented platform skip.",
    )
    args = parser.parse_args()
    if args.pytest and (
        not args.junit_dir or args.enable_retry or args.timeout_from_est_time
    ):
        parser.error(
            "--pytest requires --junit-dir and fixed timeouts, without retries"
        )
    if not args.pytest and (args.junit_dir or args.suite_timeout or args.allow_skip):
        parser.error("JUnit, suite budget, and skip policy require --pytest")
    if args.timeout_per_file <= 0 or (
        args.suite_timeout is not None and args.suite_timeout <= 0
    ):
        parser.error("timeouts must be positive")

    # Validate auto-partition arguments
    if (args.auto_partition_id is not None) != (args.auto_partition_size is not None):
        parser.error(
            "--auto-partition-id and --auto-partition-size must be specified together."
        )
    if args.auto_partition_size is not None:
        if args.auto_partition_size <= 0:
            parser.error("--auto-partition-size must be positive.")
        if not 0 <= args.auto_partition_id < args.auto_partition_size:
            parser.error(
                f"--auto-partition-id must be in range [0, {args.auto_partition_size}), "
                f"but got {args.auto_partition_id}"
            )

    exit_code = run_a_suite(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
