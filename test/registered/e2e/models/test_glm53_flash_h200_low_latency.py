"""H200 per-commit coverage for the GLM-5.3-Flash low latency recipe."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.glm53_flash_common import COMMON_SERVER_ARGS, _GLM53FlashH200Base
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin

register_cuda_ci(est_time=521, stage="extra-b", runner_config="8-gpu-h200")


class TestGLM53FlashH200LowLatency(
    SpecDecodingMixin,
    GSM8KMixin,
    _GLM53FlashH200Base,
):
    gsm8k_score_threshold = 0.96
    gsm8k_num_examples = 1319
    gsm8k_num_threads = 256
    gsm8k_backend = "sgl_eval"
    gsm8k_thinking = True
    gsm8k_max_tokens = 8192
    gsm8k_temperature = 1.0
    gsm8k_top_p = 0.95
    accept_length_thres = 4.0
    bs_1_speed_thres = 200
    server_args = [
        *COMMON_SERVER_ARGS,
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
    ]


if __name__ == "__main__":
    unittest.main()
