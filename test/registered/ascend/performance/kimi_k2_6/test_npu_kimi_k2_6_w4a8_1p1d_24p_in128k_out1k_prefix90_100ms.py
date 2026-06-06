import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    KIMI_K2_6_W4A8_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-pd-sep-2-node",
    nightly=True,
)

PREFILL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "60",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_BUFFSIZE": "1800",
}

DECODE_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "60",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_BUFFSIZE": "1200",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
}

PREFILL_ARGS = [
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--disaggregation-mode",
    "prefill",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.78,
    "--max-running-requests",
    8,
    "--chunked-prefill-size",
    16384,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
]

DECODE_ARGS = [
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--disaggregation-mode",
    "decode",
    "--nnodes",
    "2",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    32,
    "--mem-fraction-static",
    0.82,
    "--max-running-requests",
    32,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--enable-dp-attention",
    "--dp-size",
    4,
    "--disable-radix-cache",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    8,
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    1,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    2,
    "--speculative-draft-model-quantization",
    "unquant",
]

MODEL_CONFIG = {
    "model_path": KIMI_K2_6_W4A8_MODEL_PATH,
    "prefill_args": PREFILL_ARGS,
    "decode_args": DECODE_ARGS,
    "prefill_envs": PREFILL_ENVS,
    "decode_envs": DECODE_ENVS,
    "router_args": ["--policy", "cache_aware"],
    "router_envs": {},
}


class TestNPUKimiK2_6_W4A8_1P1D_24p_In64k_Out1k_100ms(
    TestAscendPerfMultiNodePdSepTestCaseBase
):
    """Test NPU performance for Kimi-K2.6-w4a8 1P+1D 24p: input_len=131072, output_len=1024, TPOT=100ms"""

    model_config = MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "random"
    max_concurrency = 8
    num_prompts = 8
    request_rate = 1
    aisbench_repeat_rate = 0.9
    input_len = 131072
    output_len = 1024
    random_range_ratio = 1
    tpot = 100
    ttft = 5000
    output_token_throughput = 16899

    def test_npu_kimi_k2_6_w4a8_1p1d_24p_in64k_out1k_100ms(self):
        """Run NPU performance test for 1P+1D 24p with 64k input, 1k output, TPOT=100ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
