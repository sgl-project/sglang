"""MI35x Qwen3.8-2.4T-A95B MXFP4 GSM8K accuracy + serving-perf test (8-GPU)

Tests amd/Qwen3.8-2.4T-A95B-Quark-MXFP4, AMD's day-0 Quark quantization of
Qwen/Qwen3.8-2.4T-A95B-FP8, on a single 8-GPU MI35x node.

Qwen3.8 is a 2.4T-parameter / 95B-active hybrid MoE: 23 repeats of 3 x Gated
DeltaNet -> MoE then 1 x Gated Attention -> MoE, 512 experts with 10 routed + 1
shared active. It reuses the Qwen3.5 architecture -- the checkpoint reports
``Qwen3_5MoeForCausalLM`` -- so no model code is added here. What is missing,
and what this test supplies, is nightly evidence that the ROCm kernels behind
that path keep producing correct tokens, and at what speed.

MXFP4 rather than FP8: at 2.4T parameters FP8 is ~2.4 TB against 8 x 288 GB =
2.30 TB per MI355X node, so the FP8 checkpoint has no single-node AMD recipe
(the cookbook serves it as MI300X TP8 x PP2 over two nodes) and single-node
means FP4. Only the routed experts are quantized; attention, the shared expert,
the MoE gate and ``lm_head`` stay at source precision, which is why AMD
measures the same 97.49 GSM8K as the FP8 baseline (100% recovery) and why this
test gates the FP8 checkpoint's quality even though it serves the MXFP4 one.

Both phases launch from one ``SERVER_ARGS``, which reproduces the recipe
published on the AMD model card. That is the reason accuracy and perf share a
file rather than splitting into an accuracy suite and a perf suite: the
throughput numbers then describe the exact configuration the accuracy gate
covers, and a flag change cannot drift one out from under the other. Two of
those flags are load-bearing rather than restatements of a default:

  * ``--page-size 1`` -- ``_page_size_default`` bumps the default to 64 on HIP
    when the container sets SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d, so the
    measured geometry only holds if the page size is pinned.
  * ``--attention-backend aiter`` -- no arg override picks a backend for
    ``Qwen3_5MoeForCausalLM`` on ROCm, so the AITER path has to be named.

Perf runs only once accuracy has passed (see ``accuracy_passed``): a server
that decodes garbage still benchmarks fine, so publishing its throughput would
be worse than publishing nothing.

The scorer extracts the last number in the reply and the server runs with no
``--reasoning-parser``, so a ``<think>`` block still scores: the reasoning
stays in ``message.content`` rather than being split into ``reasoning_content``,
which would leave ``content`` empty and score 0.

MXFP4 needs gfx95x, so this is MI35x-only and ROCm 7.2-only; it does not
register on gfx942 (MI300/MI325).

Registry: nightly-amd-8-gpu-mi35x-qwen38-mxfp4 suite
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import generate_simple_markdown_report
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

# Register for AMD CI - Qwen3.8 MXFP4 accuracy + perf on MI35x (~4h: the 1.2 TB
# checkpoint is loaded twice, once per phase, and dominates both)
register_amd_ci(
    est_time=14400, suite="nightly-amd-8-gpu-mi35x-qwen38-mxfp4", nightly=True
)

QWEN38_MXFP4_MODEL_PATH = os.environ.get(
    "QWEN38_MXFP4_MODEL_PATH", "amd/Qwen3.8-2.4T-A95B-Quark-MXFP4"
)
SERVER_LAUNCH_TIMEOUT = 9000
BENCH_TIMEOUT = 9000
TP_SIZE = 8
# AMD measures 0.9749 on this checkpoint. The gate sits ~5% below it, matching
# the relative tolerance the sibling Qwen3.5 MI35x evals allow.
ACCURACY_THRESHOLD = 0.93
PERF_RESULT_DIR = "performance_results_qwen38_mxfp4_mi35x"

# The AMD model card's serve recipe, shared by both phases.
SERVER_ARGS = [
    "--tp",
    str(TP_SIZE),
    "--attention-backend",
    "aiter",
    "--page-size",
    "1",
    "--chunked-prefill-size",
    "16384",
    "--mem-fraction-static",
    "0.9",
    "--trust-remote-code",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--watchdog-timeout",
    "1200",
]
# Gates the AITER MXFP4-MoE / GEMM / norm / rope kernels. The ROCm image sets
# it; a bare-pip host does not. popen_launch_server merges this over os.environ.
SERVER_ENV = {"SGLANG_USE_AITER": "1"}


class TestQwen38Mxfp4MI35x(CustomTestCase):
    """Qwen3.8-2.4T-A95B MXFP4 accuracy + serving perf for AMD MI35x."""

    # Set by the accuracy phase and read by the perf phase. unittest orders
    # methods alphabetically, so test_a_* lands before test_b_*.
    accuracy_passed = False

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_examples = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
        cls.max_tokens = int(os.environ.get("GSM8K_MAX_NEW_TOKENS", "2048"))

    def test_a_gsm8k_accuracy(self):
        """GSM8K few-shot accuracy must clear the AMD-published gate."""
        process = popen_launch_server(
            QWEN38_MXFP4_MODEL_PATH,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=SERVER_ARGS,
            env=SERVER_ENV,
        )

        try:
            requests.get(self.base_url + "/flush_cache")

            args = SimpleNamespace(
                base_url=self.base_url,
                model=QWEN38_MXFP4_MODEL_PATH,
                eval_name="gsm8k",
                num_examples=self.num_examples,
                num_threads=512,
                max_tokens=self.max_tokens,
                chat_template_kwargs={"enable_thinking": False},
            )
            metrics = run_eval(args)
            acc = metrics["score"]

            passed = acc >= ACCURACY_THRESHOLD
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  accuracy={acc:.3f} threshold={ACCURACY_THRESHOLD} {status}")

            if is_in_ci():
                summary = "### Qwen3.8-2.4T-A95B MXFP4 GSM8K (MI35x)\n\n"
                summary += "| Model | TP | Accuracy | Threshold | Status |\n"
                summary += "| ----- | -- | -------- | --------- | ------ |\n"
                summary += (
                    f"| {QWEN38_MXFP4_MODEL_PATH} | {TP_SIZE} | {acc:.3f} | "
                    f"{ACCURACY_THRESHOLD} | {status} |\n"
                )
                write_github_step_summary(summary)

            type(self).accuracy_passed = passed
            self.assertGreaterEqual(
                acc,
                ACCURACY_THRESHOLD,
                f"Qwen3.8 MXFP4 accuracy {acc:.3f} below threshold {ACCURACY_THRESHOLD}",
            )
        finally:
            kill_process_tree(process.pid)

    def test_b_serving_perf(self):
        """Serving benchmark for the configuration the accuracy phase gated."""
        if not self.accuracy_passed:
            self.skipTest(
                "GSM8K accuracy did not pass; throughput for a server that "
                "decodes incorrectly is not worth publishing"
            )

        # The leading 1 is repeated so generate_simple_markdown_report drops it
        # as a warmup: bench_one_batch_server measures every batch as it comes,
        # and batch 1 is both the first and the row a cold cache distorts most.
        batch_sizes = _parse_int_list_env("NIGHTLY_BATCH_SIZES", "1,1,8,16,64")
        input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "1024"))
        output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "1024"))

        runner = NightlyBenchmarkRunner(
            PERF_RESULT_DIR, type(self).__name__, self.base_url
        )
        runner.setup_result_directory()

        try:
            results, success = runner.run_benchmark_for_model(
                model_path=QWEN38_MXFP4_MODEL_PATH,
                batch_sizes=batch_sizes,
                input_lens=input_lens,
                output_lens=output_lens,
                other_args=SERVER_ARGS,
                variant="mxfp4",
                extra_bench_args=["--trust-remote-code"],
                timeout=SERVER_LAUNCH_TIMEOUT,
                env=SERVER_ENV,
            )[:2]
            if results:
                runner.full_report += (
                    generate_simple_markdown_report(results, "MI35x") + "\n"
                )
            self.assertTrue(
                success, f"Perf benchmark failed for {QWEN38_MXFP4_MODEL_PATH} on MI35x"
            )
        finally:
            runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
