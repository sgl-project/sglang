"""H200 per-commit CI: DeepSeek-V4-Flash FP4 Marlin (LowLatency recipe).

Launches TP=4 with Marlin FP4 MoE runner + EAGLE speculative decoding.
Runs 12 ServerSanity probes (correctness, streaming, concurrency, determinism)
plus a GSM8K accuracy gate.

Registry: base-c-test-dsv4-8-gpu-h200 (per-commit, 8x H200 — only 4 used by TP=4)
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=370, stage="base-c", runner_config="dsv4-8-gpu-h200")


def _flashinfer_has_sm90_cutlass_mxfp4() -> bool:
    try:
        from flashinfer.fused_moe import (  # noqa: F401
            interleave_moe_weights_for_sm90_mixed_gemm,
        )

        return True
    except ImportError:
        return False


MODEL = "deepseek-ai/DeepSeek-V4-Flash"
MODEL_FP8 = "sgl-project/DeepSeek-V4-Flash-FP8"
SERVER_LAUNCH_TIMEOUT = 3600
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'


class TestDSV4FlashFP4H200(
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """LowLatency recipe: TP=4, Marlin FP4, EAGLE spec decoding."""

    gsm8k_accuracy_thres = 0.93

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--moe-runner-backend",
                "marlin",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--watchdog-timeout",
                "900",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


@unittest.skipUnless(
    _flashinfer_has_sm90_cutlass_mxfp4(),
    "FlashInfer build lacks SM90 mixed-input MXFP4 helpers (PR #3084, >= 0.6.11)",
)
class TestDSV4FlashFP4H200FlashInferCutlass(
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """FlashInfer SM90 mixed-input cutlass MXFP4 backend (this PR): TP=4 + EAGLE.

    Mirrors :class:`TestDSV4FlashFP4H200` but swaps `--moe-runner-backend marlin`
    for `flashinfer_mxfp4`, exercising the SM90 cutlass path from FlashInfer PR
    #3084 end-to-end on a real DSv4-Flash checkpoint.
    """

    gsm8k_accuracy_thres = 0.93

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--moe-runner-backend",
                "flashinfer_mxfp4",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestDSV4FlashFP4NonMTPH200(
    BasicDecodeCorrectnessMixin, GSM8KMixin, CustomTestCase
):
    """LowLatency recipe without MTP: TP=4, Marlin FP4, no speculative decoding."""

    gsm8k_accuracy_thres = 0.93

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--moe-runner-backend",
                "marlin",
                "--watchdog-timeout",
                "900",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
