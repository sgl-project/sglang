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
# Benchmark: the dev's own method. Every dev scenario in Perf-1015
# (including this 128k decode shape) is measured with aisbench in
# fixed-count perf mode (dev suite Perf-1015-aisbench-fixed /
# aisbench_fixed_decode_20261017: dev C12 = 25.03 ms mean TPOT, accept
# 4.24, cache hit 99.9%). Aligning with that method requires two things
# the previous plain-aisbench attempt (run 41029888702, TPOT 44.4 ms) was
# missing:
#   1. Raw /v1/completions, NO chat template (api_type="raw" ->
#      run_aisbench.sh generates a VLLMCustomAPI model config; dev:
#      "AIS接口为 /v1/completions，不套 chat template"). The chat template
#      turns the natural-text continuation into an assistant QA task whose
#      digit-heavy output collapses DSPARK acceptance to 1.2-1.9.
#   2. Dev's warmup ordering: cache_fill (4 requests x batch 1,
#      round-robin over DP0-3, prefix materialized on every DP rank) ->
#      shape_warm (one full-shape unscored pass) -> formal single batch
#      N=C=12 with the built-in warmup disabled (aisbench defaults to 1
#      extra request). Dev's formal rounds start at a 99.9% prefix cache
#      hit; without the warmup the formal requests race their own 128k
#      prefill and decode/prefill interleaving skews TPOT.
#
# Dataset: identical natural-text rows from one GSM8K QA pool padded past
# 131072 tokens (generate_gsm8k_shared_prefix_dataset, aisbench layout) -
# the same 100%-cache shared-prefix shape the dev suite uses (its ShareGPT
# prefix is likewise one fixed text repeated for every request). The
# aisbench main loader does not dedup identical rows (run 41027224138:
# "Total data num: 12").
#
# Fallback if this regresses: benchmark_tool = BENCHSERVING +
# dataset_name = "gsm8k-shared-prefix" (commit 5a0a303708, run
# 41001026946, Mean TPOT 22.10 ms SUCCESS).
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
    # Low-latency decode tier: LONG_SEQ_ROUND 2 is what the dev 25.9 ms C12
    # TPOT was measured with (ROUND 64 is a throughput-oriented setting and
    # wrong here). No DEEPEP_HCCL_BUFFSIZE: the combine-tiling bisect is now
    # conclusive - HCCL 2000 without it captures fine (run 40820212836),
    # while HCCL 200 and HCCL 2000 both crash aclnnMoeLowLatencyCombineV2
    # EZ1008 WITH it (runs 40507429709, 40850443874). The dev bare-metal
    # 1800 value is not viable in the CI pod.
    "HCCL_BUFFSIZE": "2000",
    "DEEP_NORMAL_LONG_SEQ_ROUND": "2",
    "DEEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_USE_FIA": "1",
    # Temporarily disabled: the current image's sglang/kernel build
    # conflicts with FIAS_V2_BSND and the DSPARK draft acceptance drops to
    # zero (accept len 1.00, runs 40820212836/40864543937), which the dev
    # team confirmed. Re-enable once a fixed image ships.
    # "SGLANG_NPU_USE_FIAS_V2_BSND": "1",
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
    # Everything from here down mirrors the dev 128k-decode serve config
    # (run.sh / the 128k single-batch decode restore script): this is the
    # shape the dev 23.2-25.9 ms C12 TPOT was measured on.
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
    # C12 must land on a captured graph batch size, else decode falls back
    # to eager mode.
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
    # Dev-aligned aisbench fixed-count perf run (see the module comment):
    # raw completions, cache-fill + shape warmup, then one measured single
    # batch N=C=12.
    benchmark_tool = AISBENCHMARK
    dataset_type = "gsm8k-shared-prefix"
    dataset_path = None
    input_len = 131072
    output_len = 1024
    num_prompts = 12
    max_concurrency = 12
    request_rate = 0
    # Raw /v1/completions model config - the prompt goes out verbatim with
    # no chat template (dev Perf-1015 section 4.0).
    api_type = "raw"
    # Warmup ordering copied from the dev fixed-decode suite: 4 requests at
    # batch 1 spread the shared prefix over DP0-3, then one unscored
    # full-shape pass settles the hybrid-cache state before scoring.
    warmup_dp = 4
    warmup_shape = 1
    tpot = 30.0

    def test_npu_kimi_k3_w4a8_32p_mix_in128k_out1k_decode_prefix(self):
        """Run the 128k pure-decode low-latency TPOT acceptance test."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
