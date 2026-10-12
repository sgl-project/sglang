"""V4.1 PD disaggregation with forced retraction on the decode worker.

GPU 0-3: TP4 prefill. GPU 4-7: TP4 decode. Without DSpark, a retracted V4.1
request cannot be backed up on the host, so the prefill recomputes its prefix.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.pd_retract_kit import PDRetractKit
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import try_cached_model

# Initial estimate; update after the first B300 CI run.
register_cuda_ci(est_time=1800, stage="extra-b", runner_config="8-gpu-b300")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"

COMMON_ARGS = [
    "--language-model-only",
    "--attention-backend",
    "dsv4",
    "--moe-a2a-backend",
    "none",
    "--moe-runner-backend",
    "flashinfer_mxfp4",
    "--ep-size",
    "4",
    "--mem-fraction-static",
    "0.8",
    "--max-running-requests",
    "32",
    "--cuda-graph-max-bs-decode",
    "32",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--context-length",
    "16384",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 6}',
    "--random-seed",
    "0",
]


class TestDeepseekV41PDRetract(PDRetractKit, PDDisaggregationServerBase):
    model = MODEL
    prefill_tp_size = 4
    decode_tp_size = 4
    decode_base_gpu_id = 4

    gsm8k_score_threshold = 0.85
    gsm8k_num_examples = 200
    gsm8k_num_threads = 32
    gsm8k_num_shots = 8

    extra_prefill_args = COMMON_ARGS
    extra_decode_args = COMMON_ARGS

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(MODEL)
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        cls.launch_all()


if __name__ == "__main__":
    unittest.main()
