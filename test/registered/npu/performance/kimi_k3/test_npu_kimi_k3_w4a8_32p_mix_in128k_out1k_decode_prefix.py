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

# Pure-decode (low-latency tier) acceptance scenario for Kimi-K3 W4A8 32P
# (4-node mix): 131072 shared prefix in / 1024 out, C12/N12, mean
# TPOT <= 30 ms. run_aisbench.sh detects --repeat_rate 100 and switches to
# the aisbench_auto_tools_prefix toolkit: prefix_cache dataset, --prefix_test
# priming, --dp warm-up coverage.
#
# The server config is the B103-proven accuracy baseline (DSPARK draft,
# radix cache, decode graphs {1,4,16}) - this scenario MUST NOT share a
# server with the pure-prefill case: the dev pure-prefill tuning
# (mem-fraction 0.78 + chunked-prefill 65536) crashed this very server
# shape with an ACL 507034 in DeepEP dispatch (run 39917316049).
register_npu_ci(
    est_time=3600,
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
    0.72,
    "--chunked-prefill-size",
    8192,
    "--cuda-graph-bs-decode",
    1,
    4,
    16,
    "--max-running-requests",
    64,
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


class TestNPUKimiK3W4A832PMixIn128kOut1kDecodePrefix(
    TestNpuPerfMultiNodePdMixTestCaseBase
):
    """Pure decode low latency: 128k shared prefix in / 1024 out, C12,
    mean TPOT <= 30 ms (dev measured 23.2-25.9 ms at C12 on this shape)."""

    model_config = KIMI_K3_W4A8_32P_MODEL_CONFIG
    benchmark_tool = AISBENCHMARK
    dataset_type = AISBENCHMARK_DATASET_GSM8K
    # Must be None so run_aisbench() generates the dataset; the shared-prefix
    # regeneration happens inside run_aisbench.sh (--repeat_rate 100).
    dataset_path = None
    input_len = 131072
    output_len = 1024
    num_prompts = 12
    max_concurrency = 12
    random_range_ratio = 1
    request_rate = 0
    repeat_rate = "100"
    # Prefix warm-up coverage: forward --dp to aisbench_test.py so the
    # shared prefix is primed on all 4 DP ranks before the measured pass.
    dp = 4
    tpot = 30.0

    def test_npu_kimi_k3_w4a8_32p_mix_in128k_out1k_decode_prefix(self):
        """Run the 128k pure-decode low-latency TPOT acceptance test."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
