import logging
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHSERVING,
    KIMI_K3_DSPARK_MODEL_PATH,
    KIMI_K3_W4A8_MODEL_PATH,
    TestNpuPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

logger = logging.getLogger(__name__)

# Pure-decode (low-latency tier) acceptance scenario for Kimi-K3 W4A8 32P
# (4-node mix): 131072 shared prefix in / 1024 out, C12/N12, mean
# TPOT <= 30 ms.
#
# Benchmark: the framework's built-in sglang.bench_serving path with the
# generated-shared-prefix (GSP) dataset - the same tool and shape the dev
# acceptance runs use (test-decode_prefix.sh). The earlier aisbench route
# (run_aisbench.sh + the aisbench_auto_tools_prefix toolkit) never produced
# a valid C12 measurement: with --repeat_rate 100 every generated request is
# byte-identical, the OpenCompass dataset loader dedups them, and only 4
# unique requests ever ran (run 40431214784: TPOT 41.4-45.8 ms at an
# effective C4). The aisbench route also pulled in a fragile dependency
# chain (venv install on a shared PVC, toolkit config patching, tokenizer
# trust_remote_code); bench_serving needs none of that.
#
# GSP parameters: 1 group of 12 prompts sharing one system prompt of
# 95% x 131072 tokens; the remaining 5% is a per-request question, so the
# requests are distinct (no dedup) while the prefix still dominates. The
# 4 warmup requests (first GSP prompt, output capped at 32 tokens) are sent
# at C4 and prime the shared prefix on all 4 DP ranks via round-robin;
# pop_sglang_is_in_ci_for_gsp keeps SGLANG_IS_IN_CI out of the bench
# environment so the warmup prefix cache is NOT flushed before the
# measured pass.
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
    benchmark_tool = BENCHSERVING
    dataset_name = "generated-shared-prefix"
    dataset_path = None
    input_len = 131072
    output_len = 1024
    num_prompts = 12
    max_concurrency = 12
    random_range_ratio = 1
    request_rate = 0
    # 95% of the input is the shared system prompt; the rest is a distinct
    # per-request question (dev's 100-cache dataset is the same shape: one
    # long natural prefix + a short unique tail per request).
    repeat_rate = 0.95
    # One warmup request per DP rank primes the shared prefix everywhere
    # before the measured pass (bench_serving caps warmup output at 32).
    warmup_requests = 4
    pop_sglang_is_in_ci_for_gsp = True
    tpot = 30.0

    def test_npu_kimi_k3_w4a8_32p_mix_in128k_out1k_decode_prefix(self):
        """Run the 128k pure-decode low-latency TPOT acceptance test."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
