"""End-to-end GSM8K accuracy test for DSA cache layer split (GLM-5.2).

Layer split shards the DSA GPU KV/indexer cache layers across prefill CP ranks
(``--enable-dsa-cache-layer-split``); non-owner ranks read a layer via an
owner-broadcast into a small remote scratch buffer. It only applies to PD
prefill workers running DSA prefill-CP (a unified server would decode on the
same worker, where non-owner ranks lack the full cache), so this test drives a
PD-disaggregated GLM-5.2 deployment: a layer-split prefill worker running
interleave prefill-CP + layer split, and an ordinary decode worker that receives
full cache shards via PD transfer.

Runs nightly on an 8-GPU B200 runner (prefill TP=4 + decode TP=4).
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(
    est_time=1200,
    suite="nightly-8-gpu-b200",
    nightly=True,
)


class TestGLM52DSACacheLayerSplit(PDDisaggregationServerBase, GSM8KMixin):
    model = "nvidia/GLM-5.2-NVFP4"
    extra_prefill_env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    extra_decode_env = {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}
    decode_base_gpu_id = None

    # Full GSM8K test set (1319 questions) with a tight accuracy floor.
    gsm8k_accuracy_thres = 0.935
    gsm8k_num_questions = 1319
    gsm8k_num_threads = 200
    gsm8k_num_shots = 0

    # Prefill worker: interleave prefill-CP + DSA cache layer split on 4 GPUs
    # (TP=4 -> attn_cp_size=4, so KV/indexer layers shard 4-way across CP ranks).
    extra_prefill_args = [
        "--tp",
        "4",
        "--attn-cp-size",
        "4",
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
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
    ]
    # Decode worker: ordinary local decode cache, receives full shards via PD
    # transfer.
    extra_decode_args = [
        "--tp",
        "4",
        "--dsa-decode-backend",
        "trtllm",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.85",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.launch_all()


if __name__ == "__main__":
    unittest.main()
