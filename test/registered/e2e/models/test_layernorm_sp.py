"""E2E accuracy test for --enable-layernorm-sp (Megatron LayerNorm sequence
parallelism).

Launches a Qwen3 dense model with tp=2 and the SP flag on, then checks GSM8K
accuracy stays healthy -- SP re-associates the row-parallel all-reduce as
reduce-scatter + all-gather and runs the norm/residual regions on sequence
shards, so a correct implementation matches the non-SP result within
floating-point-reordering noise. tp>1 is required for SP to engage.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=420, stage="base-b", runner_config="2-gpu-large")

# Qwen3 dense (architecture "Qwen3ForCausalLM") is the SP allowlist entry.
LAYERNORM_SP_MODEL = "Qwen/Qwen3-8B"


class TestLayerNormSPAccuracy(CustomTestCase, GSM8KMixin):
    gsm8k_score_threshold = 0.85
    gsm8k_num_examples = 200

    @classmethod
    def setUpClass(cls):
        cls.model = LAYERNORM_SP_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--tp", "2", "--enable-layernorm-sp"],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    # test_gsm8k is provided by GSM8KMixin (asserts score >= gsm8k_score_threshold).


if __name__ == "__main__":
    unittest.main(verbosity=3)
