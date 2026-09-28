"""B200 per-commit CI: DeepSeek-V4-Flash FP4 with the trtllm attention backend.

Mirrors two of the FlashMLA recipes with a uniform-FP8 KV pool and trtllm-gen
sparse MLA for decode and prefill: the spec-decoding recipe (draft extend /
target verify / multi-step backend) and the breakable-CUDA-graph DP recipe
(DP padding, graph replay refresh, mixed chunk).
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=500, stage="base-c", runner_config="4-gpu-b200")

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

_DEEPEP_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}


class TestDSV4FlashFP4B200Trtllm(
    SpecDecodingMixin,
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """LowLatency recipe: TP=4, FP4 (mxfp4), EAGLE spec decoding."""

    gsm8k_accuracy_thres = 0.93
    accept_length_thres = 2.8
    bs_1_speed_thres = 220
    # Arbitrary distinctive digits; only needs to survive tokenization intact.
    NEEDLE = "48173"

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
                "--dsv4-attn-backend",
                "trtllm",
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
                "--chunked-prefill-size",
                "4096",
                "--disable-flashinfer-autotune",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_long_prompt_chunked_prefill_recall(self):
        # The needle sits in the first chunk and the question in the last, so
        # only a correct multi-chunk _forward_trtllm_prefill can recall it.
        filler = (
            "The expedition recorded water temperature, salinity, and current "
            "speed at every station along the transect. "
        )
        prompt = (
            f"The station beacon identifier is {self.NEEDLE}.\n\n"
            + "".join(f"[Entry {i}] {filler}" for i in range(220))
            + "\n\nQ: What is the station beacon identifier? Reply with just "
            "the number.\nA:"
        )
        # Second pass extends from the radix-cached prefix instead of prefilling it.
        for label in ("cold", "cached-prefix"):
            out = self._decode_generate(
                prompt=prompt, max_new_tokens=self.sanity_max_new_tokens_short
            )
            self.assertIn(self.NEEDLE, out, f"{label}: {out!r}")


class TestDSV4FlashFP4BreakableCudaGraphB200Trtllm(
    BasicDecodeCorrectnessMixin, GSM8KMixin, CustomTestCase
):
    """BCG recipe: TP=4, DP=4, DeepEP, DP attention, mixed chunk, no spec."""

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
                "--dsv4-attn-backend",
                "trtllm",
                "--tp",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--enable-mixed-chunk",
                "--cuda-graph-backend-prefill",
                "breakable",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--chunked-prefill-size",
                "4096",
                "--cuda-graph-max-bs-prefill",
                "1024",
                "--mem-fraction-static",
                "0.80",
                "--cuda-graph-max-bs-decode",
                "16",
                "--max-running-requests",
                "128",
                "--watchdog-timeout",
                "900",
            ],
            env=_DEEPEP_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
