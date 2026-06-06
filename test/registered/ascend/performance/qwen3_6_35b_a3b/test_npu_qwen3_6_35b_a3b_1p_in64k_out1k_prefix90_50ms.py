import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEFAULT_URL_FOR_TEST,
    QWEN3_6_35B_A3B_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_6_35B_A3B_64K_PREFIX_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1600",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "20",
}

QWEN3_6_35B_A3B_64K_PREFIX_OTHER_ARGS = [
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    131072,
    "--trust-remote-code",
    "--enable-prefill-delayer",
    "--mamba-scheduler-strategy",
    "extra_buffer",
    "--max-running-requests",
    32,
    "--max-mamba-cache-size",
    32,
    "--mem-fraction-static",
    0.65,
    "--cuda-graph-bs",
    4,
    8,
    16,
    32,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
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
]


class TestNPUQwen3_6_35BA3B_1P_AIME2026(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for Qwen3.6-35B-A3B 1p on AIME2026"""

    model = QWEN3_6_35B_A3B_MODEL_PATH
    other_args = QWEN3_6_35B_A3B_64K_PREFIX_OTHER_ARGS
    envs = QWEN3_6_35B_A3B_64K_PREFIX_ENVS
    accuracy = 0.927
    datasets = ["aime26"]
    few_shot_num = 0
    eval_batch_size = 64
    generation_config = {
        "max_tokens": 65536,
        "temperature": 0.2,
        "repetition_penalty": 1.08,
    }

    @classmethod
    def tearDownClass(cls):
        pass

    def test_npu_qwen3_6_35b_a3b_1p_aime2026(self):
        """Run NPU accuracy test for Qwen3.6-35B-A3B on AIME2026"""
        self.run_accuracy()


class TestNPUQwen3_6_35BA3B_1P_In64k_Out1k_Prefix90_50ms(
    TestAscendPerformanceTestCaseBase
):
    """Test NPU performance for Qwen3.6-35B-A3B 1p in64k out1k prefix90 50ms"""

    base_url = DEFAULT_URL_FOR_TEST
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_6_35B_A3B_MODEL_PATH
    other_args = QWEN3_6_35B_A3B_64K_PREFIX_OTHER_ARGS
    envs = QWEN3_6_35B_A3B_64K_PREFIX_ENVS
    dataset_name = "random"
    max_concurrency = 32
    num_prompts = 128
    input_len = 32000
    output_len = 1000
    random_range_ratio = 1
    prefix_hit_rate = 0.9
    tpot = 50
    aisbench_request_rate = 16
    output_token_throughput = 660

    @classmethod
    def setUpClass(cls):
        pass

    def test_npu_qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms(self):
        """Run NPU performance test for Qwen3.6-35B-A3B in64k out1k prefix90 50ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
