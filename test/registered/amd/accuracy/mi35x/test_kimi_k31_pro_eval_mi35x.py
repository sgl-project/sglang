"""MI35x Kimi-K3.1-Pro GSM8K Accuracy Evaluation Test (8-GPU)

Tests moonshotai/Kimi-K3.1-Pro on MI35x (gfx950) with the cookbook's MI355X /
Unified / Low-Latency (TP8) cell, Non-Spec.

Kimi-K3.1 is Kimi-K3 with DSA sparse attention on its MLA layers. On ROCm the
server pins the DSA prefill and decode backends to the Triton sparse MLA
kernels and the KV cache to fp8_e4m3 by itself, so no attention backend is
passed. The 2.8T checkpoint is ~1.5 TB, about 194 GB of the 288 GB on each GPU.

Deviations from the cookbook cell, each on purpose:
- No --enable-amd-monokernel and no SGLANG_AMD_MONO_K31_WIDE: the Kimi-K3.1
  monokernel hooks are not on main yet (sgl-project/sglang#48427).
- --max-running-requests and --cuda-graph-max-bs-decode are 64 instead of 4, so
  the full GSM8K split finishes inside the job.
- No --max-mamba-cache-size 60: it is below the 64 running requests, so the
  server sizes the mamba cache itself.
- --watchdog-timeout 1200 is added, as in the Kimi-K3 MI35x test, so loading
  the 1.5 TB checkpoint is not mistaken for a hang.
SGLANG_ROCM_RUNTIME_RESERVE_GB is kept from the cookbook; main does not read it
yet (#48427 adds it), so it takes effect once that lands.

Eval harness: sgl-eval's gsm8k (zero-shot chat, \\boxed{} extraction,
math_verify grading) through run_sgl_eval, on the full split with thinking on,
as in the GLM-5.3 MI35x nightly. The kimi_k3 reasoning parser keeps the trace
in `reasoning_content` and the answer in `content`, which is what sgl-eval
scores. 64 threads match --max-running-requests; temperature 1.0 and top_p 0.95
with a fixed seed, so a failure is a regression rather than a reroll.

This replaces Kimi-K3 as the scheduled MI35x Kimi nightly. The K3 test files
stay registered and can still be run by hand.

Registry: nightly-amd-accuracy-8-gpu-mi35x-kimi-k31-pro suite
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.accuracy_test_runner import (
    AccuracyTestResult,
    write_accuracy_github_summary,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Register for AMD CI - Kimi K3.1 Pro accuracy test on MI35x. 1022 s on rocm10
# with the checkpoint cached; downloading the 1.5 TB checkpoint into a cold
# cache added about 80 min on the first run.
register_amd_ci(
    est_time=1200,
    suite="nightly-amd-accuracy-8-gpu-mi35x-kimi-k31-pro",
    nightly=True,
)

KIMI_K31_MODEL_PATH = os.environ.get("KIMI_K31_MODEL_PATH", "moonshotai/Kimi-K3.1-Pro")
SERVER_LAUNCH_TIMEOUT = 9000
# Measured 0.9750 on rocm10 (0 of 1319 truncated); measured - 0.05, as for the
# other sgl-eval gsm8k gates.
BASELINE_ACCURACY = 0.92
TP_SIZE = 8
MAX_RUNNING_REQUESTS = 64

COOKBOOK_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    str(TP_SIZE),
    "--mem-fraction-static",
    "0.85",
    "--chunked-prefill-size",
    "16384",
    "--max-running-requests",
    str(MAX_RUNNING_REQUESTS),
    "--cuda-graph-max-bs-decode",
    str(MAX_RUNNING_REQUESTS),
    "--schedule-policy",
    "lpm",
    "--prefill-decode-interval",
    "16",
    "--mamba-radix-cache-strategy",
    "extra_buffer_lazy",
    "--cuda-graph-backend-prefill",
    "breakable",
    "--cuda-graph-bs-prefill",
    "256",
    "512",
    "768",
    "1024",
    "1536",
    "2048",
    "3072",
    "4096",
    "--weight-loader-prefetch-checkpoints",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--reasoning-parser",
    "kimi_k3",
    "--tool-call-parser",
    "kimi_k3",
    "--watchdog-timeout",
    "1200",
]

COOKBOOK_ENV = {
    # SGLANG_AITER_K3_OPT is gated behind SGLANG_USE_AITER, so both must be
    # set. AITER_SITUV2_A8W4 selects the W4A8 SiTU expert kernels.
    # AITER_FLYDSL_FORCE is read by AITER itself, not by sglang.
    "SGLANG_USE_AITER": "1",
    "SGLANG_AITER_K3_OPT": "1",
    "AITER_FLYDSL_FORCE": "1",
    "AITER_SITUV2_A8W4": "1",
    "PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.95",
    "SGLANG_ROCM_RUNTIME_RESERVE_GB": "6",
    "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1",
    "HSA_USERPTR_FOR_PAGED_MEM": "0",
}


class TestKimiK31ProEvalMI35x(CustomTestCase):
    """Kimi-K3.1-Pro GSM8K Accuracy Evaluation Test for AMD MI35x."""

    def test_kimi_k31_pro_gsm8k_accuracy(self):
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            KIMI_K31_MODEL_PATH,
            base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=COOKBOOK_ARGS,
            env={**os.environ, **COOKBOOK_ENV},
        )
        try:
            metrics = run_sgl_eval(
                SimpleNamespace(
                    base_url=base_url,
                    model=KIMI_K31_MODEL_PATH,
                    eval_name="gsm8k",
                    num_examples=None,
                    num_threads=MAX_RUNNING_REQUESTS,
                    max_tokens=32768,
                    temperature=1.0,
                    top_p=0.95,
                    seed=42,
                    sgl_eval_thinking=True,
                )
            )
        finally:
            kill_process_tree(process.pid)

        score = metrics["score"]
        passed = score >= BASELINE_ACCURACY
        write_accuracy_github_summary(
            "Kimi-K3.1-Pro (MI35x)",
            "gsm8k",
            [
                AccuracyTestResult(
                    model=KIMI_K31_MODEL_PATH,
                    dataset="gsm8k",
                    passed=passed,
                    score=score,
                    baseline_accuracy=BASELINE_ACCURACY,
                    error=None if passed else "below baseline",
                    latency=metrics.get("latency"),
                    variant=f"TP{TP_SIZE}",
                )
            ],
        )
        self.assertGreaterEqual(score, BASELINE_ACCURACY)


if __name__ == "__main__":
    unittest.main()
