import unittest

from test_ascend_single_mix_utils import TestSingleNodeTestCaseBase, NIC_NAME

QWEN3_235B_MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8"

QWEN3_235B_A22B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Eagle3"

QWEN3_235B_ENVS = {
    # "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "2100",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
}

QWEN3_235B_OTHER_ARGS = (
    [
        "--trust-remote-code",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "modelslim",
        "--max-running-requests",
        "480",
        "--context-length",
        "8192",
        "--dtype",
        "bfloat16",
        "--chunked-prefill-size",
        "-1",
        "--max-prefill-tokens",
        "4096",
        "--speculative-draft-model-quantization",
        "unquant",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_235B_A22B_EAGLE_MODEL_PATH,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--disable-radix-cache",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--tp",
        "16",
        "--dp-size",
        "16",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mem-fraction-static",
        "0.75",
        "--cuda-graph-bs",
        "6",
        "8",
        "10",
        "12",
        "15",
        "18",
        "28",
        "30",
    ]
)

class TestQwen3_235B(TestSingleNodeTestCaseBase):
    model = QWEN3_235B_MODEL_PATH
    other_args = QWEN3_235B_OTHER_ARGS
    envs = QWEN3_235B_ENVS
    dataset_name = "random"
    max_concurrency = 480
    num_prompts = 480
    input_len = 2048
    output_len = 2048
    random_range_ratio = 1
    tpot = 43.3
    # T: 205@50ms.   800I: 1.8*T
    output_token_throughput = 5787

    def test_qwen3_235b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
