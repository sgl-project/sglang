import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyMultiNodePdMixTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    KIMI_K2_6_W4A8_MODEL_PATH,
    TestAscendPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-8-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
    "HCCL_BUFFSIZE": "2400",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "SGLANG_NPU_USE_MLAPO": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_SET_CPU_AFFINITY": "1",
}

OTHER_ARGS = [
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
    32,
    "--nnodes",
    2,
    "--mem-fraction-static",
    0.62,
    "--max-running-requests",
    48,
    "--chunked-prefill-size",
    65536,
    "--context-length",
    65536,
    "--max-prefill-tokens",
    65536,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--enable-dp-attention",
    "--dp-size",
    32,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    1,
    2,
    4,
    6,
    8,
    10,
    12,
    "--disable-radix-cache",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-num-steps",
    4,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    5,
    "--speculative-draft-model-quantization",
    "unquant",
]

MODEL_CONFIG = {
    "model_path": KIMI_K2_6_W4A8_MODEL_PATH,
    "other_args": OTHER_ARGS,
    "node_envs": ENVS,
}


class TestNPUKimiK2_6_W4A8_8P_AIME2025(TestAscendAccuracyMultiNodePdMixTestCaseBase):

    model_config = MODEL_CONFIG
    accuracy = 0.961
    datasets = ["aime25"]
    eval_batch_size = 64
    generation_config = {"max_tokens": 65536, "temperature": 1.0}

    def test_aime2025(self):
        self.run_accuracy()


class TestNPUKimiK2_6_W4A8_8P_In64k_Out1k_50ms(
    TestAscendPerfMultiNodePdMixTestCaseBase
):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 48
    num_prompts = 48
    request_rate = 1
    input_len = 16384
    output_len = 1024
    random_range_ratio = 1
    tpot = 100
    output_token_throughput = 1000

    def test_npu_kimi_k2_6_w4a8_8p_in64k_out1k_50ms(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
