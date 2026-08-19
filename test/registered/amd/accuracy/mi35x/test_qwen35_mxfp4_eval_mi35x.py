"""MI35x Qwen3.5-397B-A17B MXFP4 GSM8K accuracy + serving-perf test (nightly).

Covers the two AMD MXFP4 checkpoints, which differ ONLY in whether the shared
expert is quantized -- and therefore in whether the shared-expert fusion
``fuse_gate`` serving path (qwen2_moe.py ``_use_aiter`` branch + the fuse_gate
append kernel) actually runs:

  * ``Qwen3.5-397B-A17B-MXFP4``     -- shared expert EXCLUDED from quant (BF16),
    so ``can_fuse_shared_expert`` returns False -> fusion OFF -> eager path.
    This is the reference-accuracy checkpoint (gate: gsm8k > 0.91).
  * ``Qwen3.5-397B-A17B-MoE-MXFP4`` -- shared expert IS quantized, so fusion is
    ON and the aiter ``fuse_gate`` GEMV-in-kernel path runs end-to-end. No
    per-PR CI exercises this path (the kernel unit test only checks the kernel
    in isolation); here we require it to *run through* cleanly e2e.

Launch config mirrors ~/run_qwen3.5_mxfp4_perf.sh (tp=2, aiter + unified attn +
flydsl, allreduce fusion). MXFP4 is MI35x-only, so this is not registered on
mi30x.

Registry: nightly-amd-accuracy-8-gpu-mi35x-qwen35-mxfp4 suite
"""

import os
import unittest
from types import SimpleNamespace
from typing import List

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import BenchmarkResult
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _parse_int_list_env,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=14400, suite="nightly-amd-accuracy-8-gpu-mi35x-qwen35-mxfp4", nightly=True
)

# Local checkpoints on the MI35x runner; overridable for other hosts.
MXFP4_MODEL_PATH = os.environ.get(
    "QWEN35_MXFP4_MODEL_PATH", "/data/amd/Qwen3.5-397B-A17B-MXFP4"
)
MOE_MXFP4_MODEL_PATH = os.environ.get(
    "QWEN35_MOE_MXFP4_MODEL_PATH", "/data/amd/Qwen3.5-397B-A17B-MoE-MXFP4"
)

SERVER_LAUNCH_TIMEOUT = 3600
BENCH_TIMEOUT = 5400

# Accuracy gate for the reference (non-fused, BF16-shared-expert) checkpoint.
MXFP4_ACC_THRESHOLD = 0.91

# Server flags from ~/run_qwen3.5_mxfp4_perf.sh (the reference dev config).
COMMON_ARGS = [
    "--tp",
    "2",
    "--attention-backend",
    "aiter",
    "--trust-remote-code",
    "--chunked-prefill-size",
    "32768",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--watchdog-timeout",
    "1200",
    "--mem-fraction-static",
    "0.9",
    "--disable-radix-cache",
    "--enable-aiter-allreduce-fusion",
    "--max-running-requests",
    "512",
    "--page-size",
    "16",
]

COMMON_ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "AITER_FLYDSL_FORCE": "1",
}


def _run_gsm8k(
    base_url: str, model: str, num_examples: int, max_tokens: int = 2048
) -> dict:
    """Few-shot GSM8K against a running server; returns run_eval metrics.

    ``metrics["score"]`` is the accuracy in [0, 1].
    """
    requests.get(base_url + "/flush_cache")
    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="gsm8k",
        api="completion",
        max_tokens=max_tokens,
        num_examples=num_examples,
        num_threads=256,
    )
    return run_eval(args)


def _generate_perf_report(results: List[BenchmarkResult]) -> str:
    """Compact markdown perf table (skips a leading warmup duplicate)."""
    header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        header += f" ({results[0].run_name})"
    header += f" [{os.getenv('GPU_CONFIG', 'MI35x')}]"

    summary = f"### {header}\n"
    summary += "| batch size | input len | latency (s) | input tput (tok/s) | output tput (tok/s) | ITL (ms) |\n"
    summary += "| ---------- | --------- | ----------- | ------------------ | ------------------- | -------- |\n"

    report = (
        results[1:]
        if len(results) > 1 and results[0].batch_size == results[1].batch_size
        else results
    )
    for r in report:
        itl = 1 / (r.output_throughput / r.batch_size) * 1000
        summary += (
            f"| {r.batch_size} | {r.input_len} | {r.latency:.2f} | "
            f"{r.input_throughput:.2f} | {r.output_throughput:.2f} | {itl:.2f} |\n"
        )
    return summary


def _run_perf(model_path: str, variant: str, base_url: str) -> None:
    """Run the nightly serving benchmark for ``model_path`` with COMMON_ENV set."""
    batch_sizes = _parse_int_list_env("NIGHTLY_BATCH_SIZES", "1,8,16,64")
    input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
    output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

    runner = NightlyBenchmarkRunner(
        f"performance_profiles_{variant}", variant, base_url
    )
    runner.setup_profile_directory()
    runner.full_report = f"## {variant}\n"

    old_env = {}
    for key, value in COMMON_ENV.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        results, success = runner.run_benchmark_for_model(
            model_path=model_path,
            batch_sizes=batch_sizes,
            input_lens=input_lens,
            output_lens=output_lens,
            other_args=COMMON_ARGS,
            variant=variant,
            extra_bench_args=["--trust-remote-code"],
            enable_profile=False,
            timeout=BENCH_TIMEOUT,
        )[:2]
        if results:
            runner.full_report += _generate_perf_report(results) + "\n"
        assert success, f"Perf benchmark failed for {model_path} on MI35x"
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        runner.write_final_report()


class TestQwen35Mxfp4MI35x(CustomTestCase):
    """Non-fused reference checkpoint (shared expert BF16): accuracy-gated."""

    base_url = DEFAULT_URL_FOR_TEST

    def test_a_gsm8k(self):
        """GSM8K accuracy must clear the reference gate (> 0.91)."""
        env = os.environ.copy()
        env.update(COMMON_ENV)
        process = popen_launch_server(
            MXFP4_MODEL_PATH,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=COMMON_ARGS,
            env=env,
        )
        try:
            metrics = _run_gsm8k(self.base_url, MXFP4_MODEL_PATH, num_examples=1319)
            print(f"[{MXFP4_MODEL_PATH}] {metrics=}")
            if is_in_ci():
                write_github_step_summary(
                    f"### gsm8k accuracy ({MXFP4_MODEL_PATH})\n"
                    f'score={metrics["score"]:.3f} '
                    f"(threshold {MXFP4_ACC_THRESHOLD})\n"
                )
            self.assertGreater(metrics["score"], MXFP4_ACC_THRESHOLD)
        finally:
            kill_process_tree(process.pid)

    def test_b_perf(self):
        """Serving performance benchmark."""
        _run_perf(MXFP4_MODEL_PATH, "qwen35-mxfp4-mi35x", self.base_url)


class TestQwen35MoeMxfp4MI35x(CustomTestCase):
    """Fused checkpoint (shared expert MXFP4 -> fuse_gate path): run-through smoke."""

    base_url = DEFAULT_URL_FOR_TEST

    def test_a_gsm8k_runthrough(self):
        """Exercise the aiter fuse_gate serving path e2e; require it to run and
        produce valid (non-degenerate) output. No strict accuracy gate -- the
        point is that the fuse_gate path serves cleanly end-to-end (the coverage
        gap no per-PR CI test fills)."""
        env = os.environ.copy()
        env.update(COMMON_ENV)
        process = popen_launch_server(
            MOE_MXFP4_MODEL_PATH,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=COMMON_ARGS,
            env=env,
        )
        try:
            metrics = _run_gsm8k(self.base_url, MOE_MXFP4_MODEL_PATH, num_examples=200)
            print(f"[{MOE_MXFP4_MODEL_PATH}] {metrics=}")
            if is_in_ci():
                write_github_step_summary(
                    f"### gsm8k run-through ({MOE_MXFP4_MODEL_PATH}, fuse_gate)\n"
                    f'score={metrics["score"]:.3f} (run-through, no gate)\n'
                )
            # Ran e2e and returned parseable answers -> fuse_gate path is healthy.
            self.assertGreater(metrics["score"], 0.0)
        finally:
            kill_process_tree(process.pid)

    def test_b_perf(self):
        """Serving performance benchmark for the fused checkpoint."""
        _run_perf(MOE_MXFP4_MODEL_PATH, "qwen35-moe-mxfp4-mi35x", self.base_url)


if __name__ == "__main__":
    unittest.main()
