import unittest

from sglang.test.ascend.performance.test_ascend_performance_utils import (
    TestMultiNodePdMixTestCaseBase,
    NIC_NAME,
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    QWEN3_235B_W8A8_MODEL_PATH
)

MODEL_CONFIG = {
    "model_path": QWEN3_235B_W8A8_MODEL_PATH,
    "node_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "HCCL_BUFFSIZE": "1600",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
    },
    "other_args": [
        "--trust-remote-code",
        "--nnodes", "2",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--quantization", "modelslim",
        "--max-running-requests", 768,
        "--context-length", 8192,
        "--dtype", "bfloat16",
        "--chunked-prefill-size", 131072,
        "--max-prefill-tokens", 458880,
        "--speculative-draft-model-quantization", "unquant",
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", QWEN3_235B_A22B_EAGLE_MODEL_PATH,
        "--speculative-num-steps", 3,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 4,
        "--disable-radix-cache",
        "--moe-a2a-backend", "deepep",
        "--deepep-mode", "auto",
        "--tp-size", 32,
        "--dp-size", 32,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mem-fraction-static", 0.8,
        "--cuda-graph-bs", 6, 8, 10, 12, 18, 24,
    ]
}

class TestQwen235B(TestMultiNodePdMixTestCaseBase):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 768
    num_prompts = 768
    input_len = 2048
    output_len = 2048
    random_range_ratio = 1
    tpot = 49.6
    # T: 205@50ms.   800I: 1.8*T
    output_token_throughput = 8781

    def test_qwen3_235b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
