import logging
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
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
# Benchmark: aisbench fixed-count perf run aligned with the dev method:
# raw /v1/completions (no chat template), a cache-fill warmup (4 requests
# at batch 1, one per DP rank) plus one unscored full-shape pass, then a
# single measured batch N=C=12 with the built-in warmup disabled. The
# dataset is identical natural-text rows from one GSM8K QA pool padded
# past 131072 tokens, so every request hits the same shared radix prefix
# (~99.9% cache hit, the dev shared-prefix shape).
#
# The pure-prefill and pure-decode server configs are mutually exclusive,
# so each scenario is its own case/server. This server mirrors the dev
# 128k decode serve config (DSPARK draft, radix cache, decode graphs
# {2,4,8,12,16}).
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
    "TRITON_CACHE_DIR": "/root/.cache/sglang/triton",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FOLDED_PROPOSAL": "0",
    "SGLANG_DSPARK_FOLDED_SAMPLING": "0",
    "SGLANG_DSPARK_STACKED_CTX_KV": "0",
    "SGLANG_DSPARK_EMBED_IN_GRAPH": "0",
    "SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    # Low-latency decode tier (dev value); DEEPEP_HCCL_BUFFSIZE must stay
    # unset - setting it crashes the DeepEP combine tiling in the CI pod.
    "HCCL_BUFFSIZE": "2000",
    "DEEP_NORMAL_LONG_SEQ_ROUND": "2",
    "DEEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_USE_FIA": "1",
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
    # Everything below mirrors the dev 128k decode serve config.
    "--mamba-ssm-dtype",
    "bfloat16",
    "--mem-fraction-static",
    0.75,
    "--max-total-tokens",
    250000,
    "--max-mamba-cache-size",
    240,
    "--chunked-prefill-size",
    20480,
    # C12 must land on a captured decode graph batch size.
    "--cuda-graph-bs-decode",
    2,
    4,
    8,
    12,
    16,
    "--max-running-requests",
    64,
    "--reasoning-parser",
    "kimi_k3",
    "--enable-cache-report",
    "--random-seed",
    540332093,
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
    # Dev-aligned aisbench fixed-count perf run (see the module comment).
    benchmark_tool = AISBENCHMARK
    dataset_type = "gsm8k-shared-prefix"
    dataset_path = None
    input_len = 131072
    output_len = 1024
    num_prompts = 12
    max_concurrency = 12
    request_rate = 0
    # Raw /v1/completions: the prompt goes out verbatim, no chat template.
    api_type = "raw"
    # Dev warmup ordering: prime the prefix on every DP rank, then one
    # unscored full-shape pass, before the measured batch.
    warmup_dp = 4
    warmup_shape = 1
    tpot = 30.0

    def test_npu_kimi_k3_w4a8_32p_mix_in128k_out1k_decode_prefix(self):
        """Run the 128k pure-decode low-latency TPOT acceptance test."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
