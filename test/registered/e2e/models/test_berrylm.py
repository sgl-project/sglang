"""BerryLM-OS: hybrid MoE with KDA gated delta-net linear attention and block
attention residuals (1 GPU, GSM8K accuracy through the default server fixture)."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

# 1-gpu-large: the runner class of the other ~18B-parameter model tests.
register_cuda_ci(
    est_time=600,
    disabled="rwb-ai/BerryLM-OS is not public yet",
    stage="extra-a",
    runner_config="1-gpu-large",
)

BERRYLM_MODEL = "rwb-ai/BerryLM-OS"


class TestBerryLM(GSM8KMixin, DefaultServerBase):
    model = BERRYLM_MODEL
    # Thinking is on by default: scored through the chat API with the reasoning
    # parser, as the other reasoning-model tests do (the completion backend's
    # 512-token cap cuts the reasoning off).
    gsm8k_backend = "sgl_eval"
    gsm8k_thinking = True
    gsm8k_num_examples = 200
    gsm8k_num_threads = 32
    gsm8k_max_tokens = 16384
    gsm8k_score_threshold = 0.90  # 0.95 on the release checkpoint
    other_args = [
        "--tp-size",
        "1",
        "--reasoning-parser",
        "berrylm",
        "--tool-call-parser",
        "berrylm",
    ]


if __name__ == "__main__":
    unittest.main()
