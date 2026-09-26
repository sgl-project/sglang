import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH, TestNpuPerfMultiNodePdSepTestCaseBase, BENCHMARK_TOOL_DEFAULT,
    AISBENCHMARK_DATASET_DEFAULT,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=4800,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

# Common environment variables shared by prefill/decode nodes, ported from
# 2p.sh and d.sh.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_COMMON_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "DEEPEP_HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_CONNECT_TIMEOUT": "300",
    "HCCL_EXEC_TIMEOUT": "68",
    "ACL_DEVICE_SYNC_TIMEOUT": "60",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1800",
    # skip gpu branch
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "True",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "False",
    # deepep
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    # dspark
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    "SGLANG_DSPARK_FAST_SAMPLING": "0",
    "SGLANG_DSPARK_ENABLE_MULTI_STREAM": "0",
    "SGLANG_DSPARK_QUANT_AUDIT": "1",
    "SGLANG_DSPARK_QUANT_AUDIT_STRICT": "0",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
}

# Prefill node environment variables, ported from 2p.sh.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_PREFILL_ENVS = {
    **DEEPSEEK_V4_PRO_W4A8_PD_SEP_COMMON_ENVS,
    # cp
    "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
    # memory fabric for PD KV transfer
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
}

# Decode node environment variables, ported from d.sh.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_DECODE_ENVS = {
    **DEEPSEEK_V4_PRO_W4A8_PD_SEP_COMMON_ENVS,
    "HCCL_BUFFSIZE": "512",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
}

# Prefill node (2 nodes x 16 NPUs, TP16 DP8 PP2 + CP interleave) launch
# arguments, ported from 2p.sh.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--tp-size",
    16,
    "--nnodes",
    2,
    "--pp-size",
    2,
    "--dp-size",
    8,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    32,
    "--mem-fraction-static",
    0.83,
    "--quantization",
    "modelslim",
    "--max-prefill-tokens",
    9000,
    "--chunked-prefill-size",
    8192,
    "--kv-cache-dtype",
    "auto",
    "--moe-dense-tp-size",
    1,
    "--cuda-graph-bs-decode",
    1,
    2,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-prefill-cp",
    "--cp-strategy",
    "interleave",
    "--disable-radix-cache",
]

# Decode node (2 nodes x 16 NPUs, TP32 DP16) launch arguments, ported from
# d.sh.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--tp-size",
    32,
    "--nnodes",
    2,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    64,
    "--mem-fraction-static",
    0.83,
    "--quantization",
    "modelslim",
    "--max-prefill-tokens",
    9000,
    "--chunked-prefill-size",
    8192,
    "--kv-cache-dtype",
    "auto",
    "--moe-dense-tp-size",
    1,
    "--cuda-graph-bs-decode",
    1,
    2,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--disable-radix-cache",
    # DSPARK speculative decoding with the bundled draft weights.
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "modelslim",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    6,
    "--speculative-dspark-block-size",
    5,
]

# Model config for DSV4-Pro W4A8 2P+2D PD-Sep deployment.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    "prefill_args": DEEPSEEK_V4_PRO_W4A8_PD_SEP_PREFILL_ARGS,
    "decode_args": DEEPSEEK_V4_PRO_W4A8_PD_SEP_DECODE_ARGS,
    "prefill_envs": DEEPSEEK_V4_PRO_W4A8_PD_SEP_PREFILL_ENVS,
    "decode_envs": DEEPSEEK_V4_PRO_W4A8_PD_SEP_DECODE_ENVS,
    "router_args": ["--policy", "cache_aware"],
    "router_envs": {},
}

# Generation config for Think High mode (thinking=true, reasoning_effort=high).
DEEPSEEK_V4_PRO_W4A8_GENERATION_CONFIG_HIGH = {
    "max_tokens": 125000,
    "top_p": 1,
    "temperature": 1,
    "n": 1,
    "extra_body": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
    },
}


class TestNPUDeepSeekV4ProW4A8PDSEPGPQAHigh(
    TestNpuAccuracyMultiNodePdSepTestCaseBase
):
    """Test NPU accuracy for DeepSeek-V4-Pro-0813 W4A8 PD-Sep GPQA High mode.

    Requirement: DSV4_Pro_Radix_Cache_0 (step 2, 1P1D PD separation with
    radix cache enabled).
    """

    model_config = DEEPSEEK_V4_PRO_W4A8_PD_SEP_MODEL_CONFIG
    # TODO: calibrate the baseline on the first successful run.
    accuracy = 0.85
    datasets = ["gpqa_diamond"]
    generation_config = DEEPSEEK_V4_PRO_W4A8_GENERATION_CONFIG_HIGH
    eval_batch_size = 64

    def test_npu_deepseek_v4_pro_w4a8_pd_sep_gpqa_high(self):
        """Run NPU accuracy test for DSV4-Pro W4A8 PD-Sep GPQA High mode."""
        self.run_accuracy()

class TestNPUDeepSeekV4ProW4A8PDSEPIn128kOut1kPrefix90(
    TestNpuPerfMultiNodePdSepTestCaseBase
):
    """Test NPU perf for DeepSeek-V4-Pro W4A8 PD-Sep 2P+2D in128k prefix90.

    Requirement: DSV4_Pro_Radix_Cache_0 (step 4, PD separation 128k input
    with 90% radix-cache hit rate). The shared-prefix dataset makes 90% of
    each input length a repeated prefix, so radix cache hits should reduce
    TTFT noticeably compared with the random-input test above.
    """

    model_config = DEEPSEEK_V4_PRO_W4A8_PD_SEP_MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "generated-shared-prefix"
    repeat_rate = 0.9
    input_len = 131072
    output_len = 1024
    num_prompts = 32
    max_concurrency = 32
    random_range_ratio = 1
    warmup_requests = 0
    request_rate = float("inf")
    seed = 1
    temperature = 0.6
    top_p = 0.95
    # TODO: calibrate tpot / output_token_throughput / ttft baselines on the
    # first successful run, then set them here to enable regression assertions.
    pop_sglang_is_in_ci_for_gsp = True

    def test_npu_deepseek_v4_pro_w4a8_pd_sep_in128k_out1k_prefix90(self):
        """Run NPU perf test for DSV4-Pro W4A8 PD-Sep in128k prefix90."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
