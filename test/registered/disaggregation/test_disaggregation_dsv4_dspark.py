"""H200 regression for DSpark P/D bootstrap."""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import try_cached_model

register_cuda_ci(est_time=700, stage="base-c", runner_config="8-gpu-h200")

DSV4_DSPARK_MODEL = os.getenv(
    "DEEPSEEK_V4_DSPARK_MODEL_PATH",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
)

_DSPARK_ARGS = [
    "--speculative-algorithm",
    "DSPARK",
    "--moe-runner-backend",
    "marlin",
    "--watchdog-timeout",
    "900",
]


class TestDisaggregationDSV4DSpark(PDDisaggregationServerBase, GSM8KMixin):
    prefill_tp_size = 4
    decode_tp_size = 4
    decode_base_gpu_id = 4

    gsm8k_accuracy_thres = 0.93
    gsm8k_num_questions = 200
    gsm8k_num_threads = 64

    extra_prefill_args = _DSPARK_ARGS
    extra_decode_args = _DSPARK_ARGS

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(DSV4_DSPARK_MODEL)
        cls.launch_all()


if __name__ == "__main__":
    unittest.main()
