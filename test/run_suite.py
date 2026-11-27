import argparse
import math
import os
import sys
from typing import List

import tabulate

from sglang.test.ci.ci_register import (
    CIRegistry,
    HWBackend,
    auto_partition,
    collect_tests,
    filter_tests,
    load_live_est,
    registered_test_files,
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
    "mps": HWBackend.MPS,
    "ppu": HWBackend.PPU,
    "mlu": HWBackend.MLU,
}

# Per-commit test suites. On a PR, NVIDIA and AMD base suites run with `run-ci`;
# their extra-* suites and every other hardware's suites also need that
# hardware's `run-ci-<hw>-extra` label (scripts/ci/opt_in_ci_tiers.json).
PER_COMMIT_SUITES = {
    HWBackend.CPU: [
        "base-a-test-cpu",
        "extra-test-cpu-intel",
        "extra-tp-test-cpu-intel",
        "extra-test-cpu-arm64",
    ],
    HWBackend.AMD: [
        "base-a-test-1-gpu-small-amd",
        "base-b-test-1-gpu-small-amd",
        "base-b-test-1-gpu-small-amd-mi35x",
        "base-b-disaggregation-test-large-8-gpu-amd-mi35x",
        "base-b-test-1-gpu-large-amd",
        "base-b-test-2-gpu-large-amd",
        "base-a-jit-kernel-unit-test-amd",
        "base-a-jit-kernel-benchmark-test-amd",
        "base-a-sgl-kernel-unit-test-2-gpu-amd",
        "base-c-test-4-gpu-amd",
        "base-c-test-large-8-gpu-amd",
        "base-c-test-large-8-gpu-amd-mi35x",
        # extra-a: label-gated PR opt-in suites in pr-test-amd-extra.yml
        # (mirror of CUDA extra-a; tests stay tagged per-commit but only
        # dispatch when the PR carries the `run-ci-amd-extra` label). 1-gpu-small
        # carries the mock-model / kv_canary unit + single-GPU canary e2e
        # tests; 1-gpu-large carries the subset of model e2e tests validated
        # to pass on mi325 (quant fp8kv-triton, sessions streaming-session
        # EAGLE3, spec standalone triton-backend variant); 2-gpu-large carries
        # the multi-GPU (TP/PP/PD) mock-model + kv_canary e2e tests. The rest
        # of CUDA extra-a tests fail on ROCm (missing flash_attn.cute/flash_ops
        # kernels, OOM, or accuracy regressions — e.g. gemma4-mtp-31b dips
        # below the gsm8k floor on the topk=3 leg) and stay CUDA-only for now.
        # Tests that pass on ROCm but sit in CUDA's extra tier (basic_perf,
        # torch.compile, LoRA/RL weight updates, ...) are registered here too
        # rather than in the always-on base-b/c suites.
        "extra-a-test-1-gpu-small-amd",
        "extra-a-test-1-gpu-large-amd",
        "extra-a-test-2-gpu-large-amd",
        # extra-b: 4-GPU tests that CUDA runs in extra-b (4-gpu-h100/b200).
        "extra-b-test-4-gpu-amd",
    ],
    HWBackend.MUSA: [],
    HWBackend.CUDA: [
        "base-a-test-1-gpu-small",
        "base-b-test-1-gpu-small",
        "base-b-test-1-gpu-large",
        "base-b-test-2-gpu-large",
        "base-b-test-4-gpu-b200",
        "base-b-kernel-unit-test-1-gpu-small",
        "base-b-kernel-unit-test-1-gpu-large",
        "base-b-kernel-unit-test-4-gpu-b200",
        "base-b-kernel-benchmark-test-1-gpu-small",
        "base-b-kernel-benchmark-test-1-gpu-large",
        "base-b-kernel-benchmark-test-1-gpu-b200",
        # Diffusion keeps case-level pytest partitioning behind registered
        # bridge files while sharing this discovery and dispatch entry point.
        "base-b-diffusion-test-1-gpu-large",
        "base-b-diffusion-test-1-gpu-small",
        "base-b-diffusion-test-4-gpu-b200",
        "base-b-diffusion-test-2-gpu-large",
        "base-b-diffusion-unit-test-1-gpu-large",
        "base-b-diffusion-unit-test-1-gpu-b300",
        "base-c-test-4-gpu-h100",
        "base-c-test-4-gpu-b200",
        "base-c-test-2-gpu-gb300",
        "base-c-test-4-gpu-gb300",
        "base-c-test-8-gpu-b300",
        # extra-a / extra-b: label-gated PR opt-in suites in pr-test-extra.yml
        # (tests still tagged per-commit but skipped on default PR runs).
        "extra-a-test-1-gpu-small",
        "extra-a-test-1-gpu-large",
        "extra-a-test-2-gpu-large",
        "extra-b-test-4-gpu-h100",
        "extra-b-test-4-gpu-b200",
        "extra-b-test-8-gpu-h200",
        "extra-b-test-8-gpu-h20",
        "extra-b-test-8-gpu-b300",
        "extra-b-test-2-gpu-gb300",
        "extra-b-test-4-gpu-gb300",
    ],
    HWBackend.NPU: [
        "extra-a-test-npu",
        "extra-b-test-1-npu-a3",
        "extra-b-test-2-npu-a3",
        "extra-b-test-4-npu-a3",
        "extra-b-test-8-npu-a3",
        "extra-b-test-16-npu-a3",
        "extra-c-acc-test-2-npu-a3",
        "extra-c-acc-test-16-npu-a3",
        "extra-c-perf-test-2-npu-a3",
        "extra-c-perf-test-16-npu-a3",
    ],
    HWBackend.XPU: [
        "extra-test-1-gpu-xpu",
    ],
    HWBackend.MLX: [
        "extra-unit-test-mlx",
        "extra-e2e-test-mlx",
    ],
    HWBackend.MPS: [
        "extra-unit-test-mps",
        # extra-e2e needs a self-hosted Apple Silicon runner; dispatched manually.
        "extra-e2e-test-mps",
    ],
    HWBackend.PPU: [
        "extra-test-1-gpu-ppu",
    ],
    HWBackend.MLU: [
        "extra-test-1-mlu",
        "extra-test-2-mlu",
    ],
}

# Nightly test suites (run nightly, organized by GPU configuration)
NIGHTLY_SUITES = {
    HWBackend.CUDA: [
        # `stage="nightly"` + a runner_config, same `{stage}-test-{runner_config}`
        # shape as the per-commit suites. No `nightly=True`: the stage name
        # carries the cadence; only the legacy suites below still need the flag.
        "nightly-test-1-gpu-small",
        "nightly-test-1-gpu-large",
        "nightly-test-2-gpu-large",
        "nightly-test-4-gpu-h100",
        "nightly-test-4-gpu-b200",
        "nightly-test-4-gpu-gb300",
        "nightly-test-8-gpu-h200",
        "nightly-test-8-gpu-b200",
        "nightly-test-8-gpu-b300",
    ],
    HWBackend.AMD: [
        "nightly-test-amd",
        "nightly-test-1-gpu-amd",
        "nightly-kernel-test-1-gpu-amd",
        "nightly-test-1-gpu-amd-mi35x",
        "nightly-test-1-gpu-amd-zimage-turbo",
        "nightly-test-2-gpu-amd",
        "nightly-test-2-gpu-amd-mi35x-deepseek-r1-mxfp4-tp2",
        "nightly-test-8-gpu-amd-mi35x-deepseek-r1-mxfp4-tp4",
        "nightly-accuracy-test-8-gpu-amd-mi35x-kimi-k31-pro",
        "nightly-accuracy-test-8-gpu-amd-mi35x-muse-spark",
        "nightly-accuracy-test-8-gpu-amd-mi35x-muse-spark-mmmu",
        "nightly-test-8-gpu-amd-mi35x-qwen38-mxfp4",
        "nightly-test-8-gpu-amd-mi35x-glm53-flash",
        "nightly-accuracy-test-8-gpu-amd-glm53",
        "nightly-test-8-gpu-amd-mi35x-glm53",
        "nightly-test-4-gpu-amd",
        "nightly-test-8-gpu-amd",
        "nightly-accuracy-test-8-gpu-amd-deepseek-v4-flash",
        "nightly-test-8-gpu-amd-mi35x-deepseek-v4-flash",
        "nightly-test-4-gpu-amd-mi35x-deepseek-v41-flash",
        # MI35x 8-GPU suite (different model configs)
        "nightly-test-8-gpu-amd-mi35x",
    ],
    HWBackend.MUSA: [
        "nightly-test-1-gpu-musa",
    ],
    HWBackend.CPU: [
        "nightly-test-cpu-intel",
    ],
    HWBackend.NPU: [
        "nightly-test-1-npu-a3",
        "nightly-test-2-npu-a3",
        "nightly-test-4-npu-a3",
        "nightly-test-8-npu-a3",
        "nightly-test-16-npu-a3",
        "nightly-acc-test-2-npu-a3",
        "nightly-acc-test-8-npu-a3",
        "nightly-acc-test-16-npu-a3",
        "nightly-acc-test-4-npu-a5",
        "nightly-perf-test-2-npu-a3",
        "nightly-perf-test-4-npu-a3",
        "nightly-perf-test-16-npu-a3",
        "full-test-1-npu-a3",
        "full-test-2-npu-a3",
        "full-test-4-npu-a3",
        "full-test-8-npu-a3",
        "full-test-16-npu-a3",
        "full-acc-test-2-npu-a3",
        "full-acc-test-4-npu-a3",
        "full-acc-test-8-npu-a3",
        "full-acc-test-16-npu-a3",
        "full-perf-test-2-npu-a3",
        "full-perf-test-4-npu-a3",
        "full-perf-test-8-npu-a3",
        "full-perf-test-16-npu-a3",
    ],
    HWBackend.XPU: [
        "nightly-kernel-main-test-1-gpu-xpu",
        "nightly-kernel-main-test-2-gpu-xpu",
        "nightly-kernel-main-test-4-gpu-xpu",
        "nightly-kernel-wheel-test-1-gpu-xpu",
    ],
}


OTHER_SUITES = {
    HWBackend.CPU: [
        "default",
        # `stage="weekly"`, dispatched by weekly-test-cpu.yml.
        "weekly-test-cpu",
    ],
    HWBackend.CUDA: [
        "stress",
        # `stage="weekly"` -- same shape. The three dicts group names for
        # readability only; validation reads their union. One entry per row of
        # the matrix in weekly-test-nvidia.yml.
        "weekly-test-1-gpu-large",
        "weekly-test-2-gpu-large",
        "weekly-test-4-gpu-h100",
        "weekly-test-4-gpu-b200",
        "weekly-test-8-gpu-h200",
        "weekly-test-8-gpu-b200",
        "weekly-test-8-gpu-b300",
    ],
}


_SUITE_CHECKED_BACKENDS = {
    HWBackend.CUDA,
    HWBackend.CPU,
    HWBackend.MUSA,
    HWBackend.XPU,
    HWBackend.MLX,
    HWBackend.MPS,
    HWBackend.PPU,
    HWBackend.MLU,
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


def warn_unknown_suites(hw: HWBackend, suites: List[str]):
    # Union of all three dicts, not just the per-commit or nightly half:
    # CUDA nightly suites are selected by name alone, without --nightly.
    valid = _valid_suites_by_backend().get(hw, set())
    for suite in suites:
        if suite not in valid:
            print(f"Warning: Unknown suite {suite} for backend {hw.name}")


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


def load_only_file_list(list_path: str, repo_root: str) -> set:
    """Read repo-relative test paths (one per line, optional ::case suffix)
    and normalize them to the absolute-path form CIRegistry.filename uses."""
    with open(list_path) as f:
        entries = [line.strip() for line in f if line.strip()]
    if not entries:
        print(f"Warning: empty only-file list: {list_path}")
    out = set()
    for entry in entries:
        file_part = entry.split("::", 1)[0].replace("\\", "/").removeprefix("./")
        if not os.path.isabs(file_part):
            file_part = os.path.join(repo_root, file_part)
        out.add(os.path.normpath(file_part))
    return out


def run_a_suite(args):
    hw = HW_MAPPING[args.hw]
    # --suite accepts a comma-separated list; the matched tests are unioned
    # into one pool so a single runner can LPT-partition several suites
    # together (e.g. base-b + base-c on one Xeon SPR box). A single suite is
    # just a one-element list, so existing callers are unaffected.
    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    nightly = args.nightly
    auto_partition_id = args.auto_partition_id
    auto_partition_size = args.auto_partition_size

    # Use absolute paths so the script works from any working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    # Strict: all discovered files must have proper registration
    sanity_check = True

    all_tests = collect_tests(
        registered_test_files(repo_root), sanity_check=sanity_check
    )
    validate_all_suites(all_tests)
    warn_unknown_suites(hw, suites)
    ci_tests, skipped_tests = filter_tests(all_tests, hw, suites, nightly)

    # Precision-test mode: keep only the coverage-recommended files before
    # partitioning, so matrix jobs shard the filtered pool (balanced and
    # disjoint across partitions). An empty intersection yields no tests,
    # which run_unittest_files treats as success.
    if args.only_file_list:
        only_files = load_only_file_list(args.only_file_list, repo_root)
        before = len(ci_tests)
        ci_tests = [t for t in ci_tests if t.filename in only_files]
        print(
            f"Precision filter (--only-file-list {args.only_file_list}): "
            f"{before} -> {len(ci_tests)} test file(s)"
        )

    if auto_partition_size:
        live_est = load_live_est(args.partition_model_file, suites, repo_root)
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
        fork_worker_batch_size=args.fork_worker_batch_size,
        fork_worker_preload=args.fork_worker_preload,
        timeout_overrides=dict(args.timeout_override),
    )


def _parse_timeout_override(value: str) -> tuple[str, float]:
    path, sep, seconds = value.rpartition("=")
    try:
        timeout = float(seconds)
    except ValueError:
        timeout = 0
    if not sep or not path or not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError(f"expected PATH=SECONDS, got {value!r}")
    return path, timeout


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
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help=(
            "Test suite to run. Accepts a comma-separated list of suites "
            "(e.g. 'base-b-test-1-gpu-small,base-b-test-1-gpu-large'); their tests are unioned "
            "into one pool before partitioning."
        ),
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        help=(
            "Include tests registered with nightly=True (AMD/NPU/XPU/MUSA). "
            "CUDA/CPU scheduled suites are selected by name and take no flag."
        ),
    )
    parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1200,
        help="The time limit for running one file in seconds (default: 1200).",
    )
    parser.add_argument(
        "--timeout-override",
        type=_parse_timeout_override,
        action="append",
        default=[],
        metavar="PATH=SECONDS",
        help="First-attempt timeout matched by path suffix; retries use --timeout-per-file. Repeatable.",
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
        "--only-file-list",
        type=str,
        default=None,
        help=(
            "File with one test path per line (repo-relative, optional ::case "
            "suffix); restrict the run to files in this list. Used by the "
            "coverage-based precision test mode; partitioning shards the "
            "filtered pool, so matrix jobs stay balanced and disjoint."
        ),
    )
    parser.add_argument(
        "--fork-worker-batch-size",
        type=int,
        default=1,
        help=(
            "Preload common modules, then run this many files in isolated fork "
            "children (default: 1, preserving one exec per file)."
        ),
    )
    parser.add_argument(
        "--fork-worker-preload",
        nargs="*",
        default=[],
        help=(
            "Extra modules the fork worker imports before forking, on top of "
            "the common numpy/torch/triton stack."
        ),
    )
    args = parser.parse_args()

    if args.fork_worker_batch_size <= 0:
        parser.error("--fork-worker-batch-size must be positive")

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
