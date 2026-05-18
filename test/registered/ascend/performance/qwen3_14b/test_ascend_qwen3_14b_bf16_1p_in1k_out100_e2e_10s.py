import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHSERVING,
    QWEN3_14B_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-2-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_14B_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "ASCEND_USE_FIA": "0",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
}

QWEN3_14B_OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--disable-radix-cache",
    "--mem-fraction-static",
    0.85,
    "--tp-size",
    2,
    "--dp-size",
    1,
    "--sampling-backend",
    "ascend",
    "--max-running-requests",
    96,
    "--served-model-name",
    "Qwen3-14B",
    "--chunked-prefill-size",
    -1,
    "--cuda-graph-bs",
    96,
]


class TestQwen14B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHSERVING
    model = QWEN3_14B_MODEL_PATH
    other_args = QWEN3_14B_OTHER_ARGS
    envs = QWEN3_14B_ENVS
    dataset_name = "random"
    backend = "sglang-oai-chat"
    num_prompts = 1000
    input_len = 1024
    output_len = 100
    random_range_ratio = 1
    request_rate = 7
    seed = 1000
    mean_e2e_latency = 10000
    output_token_throughput = 682

    def test_qwen3_14b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
