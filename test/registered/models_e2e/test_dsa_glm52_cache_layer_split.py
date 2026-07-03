"""End-to-end GSM8K accuracy test for DSA cache layer split (GLM-5.2).

Layer split shards the DSA GPU KV/indexer cache layers across prefill CP ranks
(``--enable-dsa-cache-layer-split``); non-owner ranks read a layer via an
owner-broadcast into a small remote scratch buffer. It only applies to PD
prefill workers running DSA prefill-CP, so this test drives a PD-disaggregated
GLM-5.2 deployment (layer-split prefill + ordinary decode) and asserts GSM8K
accuracy stays intact.

Modeled on ``test/registered/models_e2e/test_dsa_glm52_tp_mtp.py`` but using the
PD-disaggregation fixture because layer split is a PD-prefill-only feature.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(
    est_time=600,
    stage="base-c",
    runner_config="8-gpu-h200",
)


class TestGLM52DSACacheLayerSplit(PDDisaggregationServerBase, GSM8KMixin):
    model = "zai-org/GLM-5.2-FP8"

    gsm8k_accuracy_thres = 0.90
    gsm8k_num_questions = 200
    gsm8k_num_threads = 200
    gsm8k_num_shots = 0

    # Prefill worker: DSA prefill-CP (round-robin / interleave) + layer split.
    extra_prefill_args = [
        "--tp",
        "4",
        "--dsa-prefill-backend",
        "trtllm",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--enable-dsa-cache-layer-split",
        "--enable-dsa-prefill-context-parallel",
        "--dsa-prefill-cp-mode",
        "round-robin-split",
        "--mem-fraction-static",
        "0.85",
        "--chunked-prefill-size",
        "4096",
        "--max-prefill-tokens",
        "4096",
    ]
    # Decode worker: ordinary local decode cache, receives full shards via PD.
    extra_decode_args = [
        "--tp",
        "4",
        "--dsa-decode-backend",
        "trtllm",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.85",
        "--base-gpu-id",
        "4",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.launch_all()


if __name__ == "__main__":
    unittest.main()
