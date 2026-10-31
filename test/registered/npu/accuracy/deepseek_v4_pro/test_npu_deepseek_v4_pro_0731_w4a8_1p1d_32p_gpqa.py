import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    run_bench_serving,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=4800,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

DEEPSEEK_V4_PRO_W4A8_PD_SEP_COMMON_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "MF_HYBM_USE_VMM_SEGMENT": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
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
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    # multi-node
    "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    # dspark
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    "SGLANG_DSPARK_FAST_SAMPLING": "0",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1800",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1800",
    # PD-sep: required when prefill enables CP but decode does not
    "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
}

DEEPSEEK_V4_PRO_W4A8_PD_SEP_PREFILL_ENVS = {
    **DEEPSEEK_V4_PRO_W4A8_PD_SEP_COMMON_ENVS,
    "DEEPEP_HCCL_BUFFSIZE": "2048",
}

DEEPSEEK_V4_PRO_W4A8_PD_SEP_DECODE_ENVS = {
    **DEEPSEEK_V4_PRO_W4A8_PD_SEP_COMMON_ENVS,
    "DEEPEP_HCCL_BUFFSIZE": "1600",
    # mtp
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

DEEPSEEK_V4_PRO_W4A8_PD_SEP_PREFILL_ARGS = [
    "--nnodes",
    2,
    "--tp-size",
    16,
    "--pp-size",
    2,
    "--dp-size",
    8,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--load-balance-method",
    "total_tokens",
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
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
    0.82,
    "--quantization",
    "modelslim",
    "--max-prefill-tokens",
    70000,
    "--chunked-prefill-size",
    65536,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--disable-cuda-graph",
    "--enable-prefill-cp",
    "--cp-strategy",
    "interleave",
    "--enable-metrics",
    "--enable-cache-report",
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--hicache-ratio",
    1.5,
    "--hicache-write-policy",
    "write_through",
    "--max-total-tokens",
    120000,
]

DEEPSEEK_V4_PRO_W4A8_PD_SEP_DECODE_ARGS = [
    "--nnodes",
    2,
    "--tp-size",
    32,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--load-balance-method",
    "round_robin",
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    8,
    "--mem-fraction-static",
    0.78,
    "--quantization",
    "modelslim",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs-decode",
    1,
    2,
    8,
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
    "--enable-metrics",
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
    """Test NPU accuracy for DeepSeek-V4-Pro-0813 W4A8 32p 1P1D GPQA"""

    model_config = DEEPSEEK_V4_PRO_W4A8_PD_SEP_MODEL_CONFIG
    # (reference: Pro W4A8 GPQA Diamond baseline is 0.85).
    accuracy = 0.85
    datasets = ["gpqa_diamond"]
    generation_config = DEEPSEEK_V4_PRO_W4A8_GENERATION_CONFIG_HIGH
    eval_batch_size = 64

    def test_npu_deepseek_v4_pro_w4a8_pd_sep_gpqa_high(self):
        """Run NPU accuracy test for DSV4-Pro W4A8 PD-Sep GPQA High mode."""
        self.run_accuracy()

    def test_npu_deepseek_v4_pro_w4a8_pd_sep_l2_host_hit_rate(self):
        """run_bench_serving in 128k out 1k asserts the host hit rate"""
        run_bench_serving(
            host=self.host,
            port=self.port,
            model_path=self.model_config.get("model_path"),
            dataset_name="generated-shared-prefix",
            input_len=131072,
            output_len=1024,
            max_concurrency=8,
            gsp_num_groups=3,
            gsp_prompts_per_group=8,
            seed=100,
            warmup_requests=16,
            repeat_rate=0.9,
            cache_report=True,
            min_host_hit_rate_pct=0
        )


if __name__ == "__main__":
    unittest.main()
