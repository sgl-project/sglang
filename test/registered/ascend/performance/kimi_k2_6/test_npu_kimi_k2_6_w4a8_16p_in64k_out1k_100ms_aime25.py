import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdMixTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    KIMI_K2_6_W4A8_MODEL_PATH,
    TestNpuPerfMultiNodePdMixTestCaseBase,
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
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "HCCL_BUFFSIZE": "4400",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
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
    0.662,
    "--max-running-requests",
    32,
    "--chunked-prefill-size",
    262144,
    "--context-length",
    75000,
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
    "--disable-radix-cache",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--reasoning-parser",
    "kimi_k2",
    "--tool-call-parser",
    "kimi_k2",
]

MODEL_CONFIG = {
    "model_path": KIMI_K2_6_W4A8_MODEL_PATH,
    "other_args": OTHER_ARGS,
    "node_envs": ENVS,
}


class TestNPUKimiK2_6_W4A8_16P_AIME2025(TestNpuAccuracyMultiNodePdMixTestCaseBase):

    model_config = MODEL_CONFIG
    accuracy = 0.961
    datasets = ["aime25"]
    few_shot_num = 0
    eval_batch_size = 64
    generation_config = {"max_tokens": 65536, "temperature": 1.0}

    def test_aime2025(self):
        self.run_accuracy()


class TestNPUKimiK2_6_W4A8_16P_In64k_Out1k_100ms(TestNpuPerfMultiNodePdMixTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 32
    num_prompts = 32
    input_len = 64000
    output_len = 1000
    random_range_ratio = 1
    seed = 1
    tpot = 100
    output_token_throughput = 160

    def test_npu_kimi_k2_6_w4a8_16p_in64k_out1k_100ms(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
