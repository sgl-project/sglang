"""End-to-end GSM8K accuracy test for DSA cache layer split (GLM-5.2).

Layer split shards the DSA GPU KV/indexer cache layers across prefill CP ranks
(``--enable-dsa-cache-layer-split``); non-owner ranks read a layer via an
owner-broadcast into a small remote scratch buffer. It only applies to PD
prefill workers running DSA prefill-CP (a unified server would decode on the
same worker, where non-owner ranks lack the full cache), so this test drives a
PD-disaggregated GLM-5.2 deployment: a layer-split prefill worker running
interleave prefill-CP + layer split, and an ordinary decode worker that receives
full cache shards via PD transfer.

Sized for the 4-GPU B200 runner (prefill TP=2 + decode TP=2) rather than an
8-GPU deployment, since the 8-gpu-b200 runner is nightly-only.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(
    est_time=1200,
    stage="extra-b",
    runner_config="4-gpu-b200",
    disabled="Temporarily disabled",
)


class TestGLM52DSACacheLayerSplit(PDDisaggregationServerBase, GSM8KMixin):
    model = "nvidia/GLM-5.2-NVFP4"

    # Full GSM8K test set (1319 questions) with a tight accuracy floor.
    gsm8k_accuracy_thres = 0.935
    gsm8k_num_questions = 1319
    gsm8k_num_threads = 200
    gsm8k_num_shots = 0

    # Prefill worker: interleave prefill-CP + DSA cache layer split on 2 GPUs
    # (TP=2 -> attn_cp_size=2, so KV/indexer layers shard 2-way across CP ranks).
    extra_prefill_args = [
        "--tp",
        "2",
        "--dsa-prefill-backend",
        "trtllm",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--enable-dsa-cache-layer-split",
        "--enable-prefill-cp",
        "--cp-strategy",
        "interleave",
        "--mem-fraction-static",
        "0.85",
        "--chunked-prefill-size",
        "4096",
        "--max-prefill-tokens",
        "4096",
    ]
    # Decode worker: ordinary local decode cache on the other 2 GPUs, receives
    # full shards via PD transfer.
    extra_decode_args = [
        "--tp",
        "2",
        "--dsa-decode-backend",
        "trtllm",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.85",
        "--base-gpu-id",
        "2",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.launch_all()


if __name__ == "__main__":
    unittest.main()
