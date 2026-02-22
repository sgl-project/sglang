import unittest

from sglang.test.ascend.e2e.test_ascend_performance_utils import (
    TestAscendPerfMultiNodePdMixTestCaseBase,
    GLM_4_6_W8A8_MODEL_PATH,
)
from sglang.test.ascend.e2e.test_ascend_multi_node_utils import NIC_NAME

MODEL_CONFIG = {
    "model_path": GLM_4_6_W8A8_MODEL_PATH,
    "node_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "HCCL_BUFFSIZE": "1800",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
    },
    "other_args": [
        "--trust-remote-code",
        "--nnodes", 2,
        "--attention-backend", "ascend",
        "--device", "npu",
        "--quantization", "modelslim",
        "--max-running-requests", 256,
        "--context-length", 8192,
        "--dtype", "bfloat16",
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", 1,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 2,
        "--chunked-prefill-size", 114688,
        "--max-prefill-tokens", 458880,
        "--disable-radix-cache",
        "--moe-a2a-backend", "deepep",
        "--deepep-mode", "auto",
        "--tp-size", 32,
        "--dp-size", 4,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mem-fraction-static", 0.7,
        "--cuda-graph-bs", 32, 48, 64,
    ]
}


class TestGlm46W8A8(TestAscendPerfMultiNodePdMixTestCaseBase):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 256
    num_prompts = int(max_concurrency) * 256
    input_len = 3200
    output_len = 1000
    random_range_ratio = 1
    tpot = 50
    # T: None   800I: xxxxx.     dev：3192/16@51.19ms
    output_token_throughput = 3192

    def test_qwen3_480b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
