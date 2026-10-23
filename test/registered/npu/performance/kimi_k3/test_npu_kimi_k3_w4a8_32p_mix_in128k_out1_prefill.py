import logging
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_GSM8K,
    KIMI_K3_W4A8_MODEL_PATH,
    TestNpuPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

logger = logging.getLogger(__name__)

# Pure-prefill acceptance scenario for Kimi-K3 W4A8 32P (4-node mix):
# cold 131072 in / 1 out, C1/N1, mean TTFT <= 20 s.
#
# The server config mirrors the dev pure-prefill launch on the same
# topology (prefill.sh TP mode): no DSPARK draft, radix cache disabled,
# decode cuda graphs off, mem-fraction 0.78, chunked-prefill 65536
# (16384 per DP rank), max-total-tokens 250000. Do NOT merge this config
# with the decode scenario: run 39917316049 crashed the server with an
# ACL 507034 in the DeepEP dispatch when the pure-prefill memory tuning
# (0.78 + 65536) shared a server with the DSPARK draft + decode graphs
# required by the pure-decode tier - the two scenarios need mutually
# exclusive server configs and therefore separate test cases.
#
# Known gap: a single-stream 128k prefill on this dp-attention deployment
# reached only ~2.2k tok/s (TTFT ~58 s) vs ~7.2k tok/s on the dev CPP
# (TP16 x PP4 x DP2) topology - the C1 request lights up one DP attention
# domain only. This case keeps the 20 s SLO as the gate so the CI trend
# is tracked honestly while the prefill topology question is settled.
register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

KIMI_K3_W4A8_32P_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "1",
    "TRITON_CACHE_DIR": "/tmp/triton_cache",
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

KIMI_K3_W4A8_32P_PREFILL_ARGS = [
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
    # Pure-prefill tuning (dev prefill.sh TP mode): the memory freed by
    # dropping the DSPARK draft, the decode graphs and the radix cache is
    # what makes 0.78 + 65536 safe on this topology.
    "--mem-fraction-static",
    0.78,
    "--chunked-prefill-size",
    65536,
    "--max-total-tokens",
    250000,
    "--disable-radix-cache",
    "--cuda-graph-backend-decode",
    "disabled",
    "--max-running-requests",
    8,
    "--reasoning-parser",
    "kimi_k3",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--disable-custom-all-reduce",
    "--watchdog-timeout",
    9000,
]

KIMI_K3_W4A8_32P_MODEL_CONFIG = {
    "model_path": KIMI_K3_W4A8_MODEL_PATH,
    "other_args": KIMI_K3_W4A8_32P_PREFILL_ARGS,
    "node_envs": KIMI_K3_W4A8_32P_ENVS,
}


class TestNPUKimiK3W4A832PMixIn128kOut1Prefill(TestNpuPerfMultiNodePdMixTestCaseBase):
    """Pure prefill: cold 131072 in / 1 out, C1/N1, mean TTFT <= 20 s."""

    model_config = KIMI_K3_W4A8_32P_MODEL_CONFIG
    benchmark_tool = AISBENCHMARK
    dataset_type = AISBENCHMARK_DATASET_GSM8K
    # Must be None so run_aisbench() generates the exact-length GSM8K dataset;
    # the base-class default would silently fall back to the raw ShareGPT file.
    dataset_path = None
    input_len = 131072
    output_len = 1
    num_prompts = 1
    max_concurrency = 1
    random_range_ratio = 1
    request_rate = 0
    ttft = 20000.0

    def test_npu_kimi_k3_w4a8_32p_mix_in128k_out1_prefill(self):
        """Run the 128k cold-prefill TTFT acceptance test."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
