"""Unequal attention TP must reshard sliding-window KV: two prefill ranks write
their KV heads into each decode window page, and a wrong slice fails silently."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    configure_nixl_pd_backend,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
    try_cached_model,
)

register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-h100")

SWA_SERVER_ARGS = ["--page-size", "64", "--attention-backend", "triton"]


class TestDisaggregationSWAPrefillLargerTP(PDDisaggregationServerBase, GSM8KMixin):
    prefill_tp_size = 2
    decode_tp_size = 1
    decode_base_gpu_id = 2
    extra_prefill_args = SWA_SERVER_ARGS
    extra_decode_args = SWA_SERVER_ARGS
    # Same eval and floor as the equal-TP case in
    # test_disaggregation_decode_radix_cache_swa.py.
    gsm8k_score_threshold = 0.45
    gsm8k_num_examples = 500
    gsm8k_num_threads = 100
    gsm8k_num_shots = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE)
        configure_nixl_pd_backend(cls)
        cls.launch_all()


if __name__ == "__main__":
    unittest.main()
