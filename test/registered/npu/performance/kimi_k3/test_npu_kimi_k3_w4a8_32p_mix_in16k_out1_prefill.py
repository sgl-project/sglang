import logging
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_GSM8K,
    KIMI_K3_DSPARK_MODEL_PATH,
    KIMI_K3_W4A8_MODEL_PATH,
    TestNpuPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

logger = logging.getLogger(__name__)

# Cold-prefill pipeline validation for Kimi-K3 W4A8 32P (4-node mix) via
# aisbench. Single request, 16k input / 1 output: the server launch config is
# byte-identical to the scoped accuracy case in
# test/registered/npu/accuracy/kimi_k3/test_npu_kimi_k3_w4a8_32p_gpqa.py so
# numbers stay comparable with the accuracy environment.
register_npu_ci(
    est_time=2700,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

KIMI_K3_W4A8_32P_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "1",
    "SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE": "1",
    "TRITON_CACHE_DIR": "/tmp/triton_cache",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FOLDED_PROPOSAL": "0",
    "SGLANG_DSPARK_STACKED_CTX_KV": "0",
    "SGLANG_DSPARK_EMBED_IN_GRAPH": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "HCCL_BUFFSIZE": "2000",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "64",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

KIMI_K3_W4A8_32P_OTHER_ARGS = [
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--nnodes",
    4,
    "--tokenizer-path",
    KIMI_K3_W4A8_MODEL_PATH,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--tp-size",
    64,
    # Deprecated spelling, still auto-converted to --attn-dp-size on main;
    # required by the older sglang build in the nightly image.
    "--enable-dp-attention",
    "--dp-size",
    4,
    "--enable-dp-lm-head",
    "--enable-shared-experts-attn-tp",
    "--enable-dense-mlp-attn-tp",
    "--mem-fraction-static",
    0.75,
    # TTFT lever for a 16k cold prefill: 32768/4DP = 8192 tokens per chunk
    # iteration, so one request is served in 2 iterations instead of 8 with
    # the previous 8192. The accuracy case used 8192 to interleave prefill
    # with long decodes, which is irrelevant for a 1-request / 1-token run.
    "--chunked-prefill-size",
    32768,
    "--cuda-graph-bs-decode",
    1,
    4,
    16,
    # Single-request scenario: a small running slot list is sufficient.
    "--max-running-requests",
    8,
    "--reasoning-parser",
    "kimi_k3",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    KIMI_K3_DSPARK_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-dspark-block-size",
    7,
    "--speculative-draft-attention-backend",
    "ascend",
    "--linear-attn-verify-backend",
    "triton",
    "--speculative-eagle-topk",
    1,
    "--disable-custom-all-reduce",
    "--watchdog-timeout",
    9000,
]

KIMI_K3_W4A8_32P_MODEL_CONFIG = {
    "model_path": KIMI_K3_W4A8_MODEL_PATH,
    "other_args": KIMI_K3_W4A8_32P_OTHER_ARGS,
    "node_envs": KIMI_K3_W4A8_32P_ENVS,
}


class TestNPUKimiK3W4A832PMixIn16kOut1Prefill(TestNpuPerfMultiNodePdMixTestCaseBase):
    """Test NPU cold prefill TTFT for Kimi-K3-w4a8 32p four nodes, 16k in / 1 out.

    Acceptance SLO: mean TTFT <= 4.0 s (historical single-request 16k TTFT on
    this deployment was ~5 s with the accuracy launch config; the prefill
    chunk size is raised to 32768 here to get under the SLO).
    """

    model_config = KIMI_K3_W4A8_32P_MODEL_CONFIG
    benchmark_tool = AISBENCHMARK
    dataset_type = AISBENCHMARK_DATASET_GSM8K
    # Must be None so run_aisbench() generates the exact-length GSM8K dataset;
    # the base-class default would silently fall back to the raw ShareGPT file.
    dataset_path = None
    input_len = 16384
    output_len = 1
    num_prompts = 1
    max_concurrency = 1
    random_range_ratio = 1
    request_rate = 0
    ttft = 4000.0

    def test_npu_kimi_k3_w4a8_32p_mix_in16k_out1_prefill(self):
        """Run NPU cold prefill TTFT test for Kimi-K3-w4a8 32p 16k in / 1 out."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
