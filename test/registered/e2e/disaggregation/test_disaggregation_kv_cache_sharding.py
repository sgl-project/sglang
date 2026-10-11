"""PD accuracy with logical-page KV cache sharding on a small MLA model.

Each prefill TP rank stores only its stripe of every KV page and sends only the
pages it owns; decode pulls from both prefill ranks. A wrong owner, local page
id, or send position corrupts decode's KV and collapses GSM8K accuracy.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MLA_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=200, stage="base-c", runner_config="4-gpu-b200")

# trtllm_mla is the Blackwell MLA backend that supports KV sharding.
COMMON_ARGS = ["--attention-backend", "trtllm_mla", "--page-size", "64"]


class TestDisaggregationKVCacheSharding(PDDisaggregationServerBase, GSM8KMixin):
    model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2
    # KV sharding with the radix cache on requires the Python tree core.
    extra_prefill_env = {"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python"}
    # 256-token chunks split each 5-shot prompt, so later chunks read a
    # sharded prefix and every rank sends pages from several chunks.
    extra_prefill_args = COMMON_ARGS + [
        "--enable-kv-cache-sharding",
        "--chunked-prefill-size",
        "256",
    ]
    extra_decode_args = COMMON_ARGS
    gsm8k_score_threshold = 0.70

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.launch_all()


if __name__ == "__main__":
    unittest.main()
