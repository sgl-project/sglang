import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    MIMO_V2_5_PRO_DFLASH_MODEL_PATH,
    MIMO_V2_5_PRO_FP4_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

MIMO_V2_5_PRO_PD_SEP_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_BUFFSIZE": "300",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "ASCEND_USE_FIA": "1",
    "ASCEND_MF_TRANSFER_PROTOCOL": "device_urma",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "4096",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "0",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
    "SGLANG_NPU_USE_SCATTER_PA_KV_CACHE": "1",
}

MIMO_V2_5_PRO_PD_SEP_DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_BUFFSIZE": "300",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
    "ASCEND_MF_TRANSFER_PROTOCOL": "device_urma",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "DEEPEP_HCCL_BUFFSIZE": "1200",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_NPU_USE_SCATTER_PA_KV_CACHE": "1",
}

MIMO_V2_5_PRO_PD_SEP_PREFILL_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    8,
    "--chunked-prefill-size",
    16384,
    "--trust-remote-code",
    "--max-running-requests",
    64,
    "--mem-fraction-static",
    0.85,
    "--max-total-tokens",
    800000,
    "--swa-full-tokens-ratio",
    0.3,
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--disable-radix-cache",
]

MIMO_V2_5_PRO_PD_SEP_DECODE_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    8,
    "--trust-remote-code",
    "--max-running-requests",
    16,
    "--mem-fraction-static",
    0.9,
    "--swa-full-tokens-ratio",
    0.1,
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    8,
    12,
    16,
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--speculative-algorithm",
    "DFLASH",
    "--speculative-draft-model-path",
    MIMO_V2_5_PRO_DFLASH_MODEL_PATH,
    "--speculative-num-draft-tokens",
    8,
    "--enable-metrics",
    "--disable-radix-cache",
]

MIMO_V2_5_PRO_PD_SEP_MODEL_CONFIG = {
    "model_path": MIMO_V2_5_PRO_FP4_MODEL_PATH,
    "prefill_args": MIMO_V2_5_PRO_PD_SEP_PREFILL_ARGS,
    "decode_args": MIMO_V2_5_PRO_PD_SEP_DECODE_ARGS,
    "prefill_envs": MIMO_V2_5_PRO_PD_SEP_PREFILL_ENVS,
    "decode_envs": MIMO_V2_5_PRO_PD_SEP_DECODE_ENVS,
    "router_args": ["--health-check-interval-secs", 3600, "--mini-lb"],
    "router_envs": {},
}


class TestNPUMiMoV2_5_Pro_W4A8_1P1D_16P_A5_TTFT3s(
    TestNpuPerfMultiNodePdSepTestCaseBase
):
    """Test NPU performance for MiMo-V2.5-Pro-FP4-DFlash PD separation 1p1d 16p a5 TTFT 3s"""

    model_config = MIMO_V2_5_PRO_PD_SEP_MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "random"
    input_len = 16384
    output_len = 1
    num_prompts = 200
    random_range_ratio = 1
    request_rate = 1.1
    ttft = 3000

    def test_npu_mimo_v2_5_pro_w4a8_1p1d_16p_a5_ttft3s(self):
        """Run NPU performance test for MiMo-V2.5-Pro-FP4-DFlash PD separation TTFT 3s"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
