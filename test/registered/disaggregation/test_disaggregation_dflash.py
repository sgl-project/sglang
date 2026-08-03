"""Regression for resuming DFLASH decode after disaggregated prefill."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
)

register_cuda_ci(est_time=500, stage="base-b", runner_config="2-gpu-large")

_DFLASH_ARGS = [
    "--speculative-algorithm",
    "DFLASH",
    "--speculative-draft-model-path",
    DEFAULT_DRAFT_MODEL_DFLASH,
    "--attention-backend",
    "flashinfer",
    "--max-running-requests",
    "64",
    "--mem-fraction-static",
    "0.7",
]


class TestDisaggregationDFlash(PDDisaggregationServerBase, GSM8KMixin):
    model = DEFAULT_TARGET_MODEL_DFLASH
    gsm8k_accuracy_thres = 0.70
    gsm8k_num_questions = 200
    gsm8k_num_threads = 64

    extra_prefill_args = _DFLASH_ARGS
    extra_decode_args = _DFLASH_ARGS

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.launch_all()


if __name__ == "__main__":
    unittest.main()
