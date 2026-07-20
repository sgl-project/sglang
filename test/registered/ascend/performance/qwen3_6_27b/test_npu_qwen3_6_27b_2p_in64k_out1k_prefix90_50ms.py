import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_6_27B_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_6_27B_64K_PREFIX_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
    "GDN_ATTN_BACKEND_TRITON": "1",
}

QWEN3_6_27B_64K_PREFIX_OTHER_ARGS = [
    "--tp-size",
    4,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    32768,
    "--max-prefill-tokens",
    32768,
    "--mamba-scheduler-strategy",
    "extra_buffer",
    "--trust-remote-code",
    "--max-running-requests",
    20,
    "--max-mamba-cache-size",
    160,
    "--mem-fraction-static",
    0.82,
    "--cuda-graph-bs",
    1,
    2,
    5,
    10,
    15,
    17,
    19,
    20,
    # "--enable-prefill-delayer",
    # "--prefill-delayer-queue-min-ratio",
    # 0.7,
    # "--prefill-delayer-max-delay-ms",
    # 20000,
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
]


class TestNPUQwen3_6_27B_1P_In64k_Out1k_Prefix90_50ms(TestNpuPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.6-27B 1p in64k out1k prefix90 50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = QWEN3_6_27B_MODEL_PATH
    other_args = QWEN3_6_27B_64K_PREFIX_OTHER_ARGS
    envs = QWEN3_6_27B_64K_PREFIX_ENVS
    dataset_name = "generated-shared-prefix"
    max_concurrency = 20
    num_prompts = 20
    input_len = 64000
    output_len = 1000
    random_range_ratio = 1
    seed = 1
    repeat_rate = 0.9
    request_rate = float("inf")
    warmup_requests = 1
    tpot = 50
    output_token_throughput = 225
    pop_sglang_is_in_ci_for_gsp = True

    def test_npu_qwen3_6_27b_2p_in64k_out1k_prefix90_50ms(self):
        """Run NPU performance test for Qwen3.6-27B in64k out1k prefix90 50ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
