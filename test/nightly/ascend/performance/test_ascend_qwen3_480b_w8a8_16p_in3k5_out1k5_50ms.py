import unittest

from test_ascend_multi_mix_utils import TestMultiMixUtils
from test_ascend_single_mix_utils import NIC_NAME

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot"

MODEL_CONFIG = {
    "model_path": MODEL_PATH,
    "node_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_DP_ROUND_ROBIN": "1",
        "HCCL_BUFFSIZE": "2100",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "STREAMS_PER_DEVICE": "32",
        "TASK_QUEUE_ENABLE": "0",
        "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    },
    "other_args": [
        "--trust-remote-code",
        "--nnodes",
        "2",
        "--tp-size",
        "32",
        "--dp-size",
        "32",
        "--mem-fraction-static",
        "0.75",
        "--max-running-requests",
        "384",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "modelslim",
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--cuda-graph-bs",
        6,
        8,
        10,
        12,
        "--enable-dp-lm-head",
        "--disable-cuda-graph",
        "--chunked-prefill-size",
        "-1",
        "--max-prefill-tokens",
        "7168",
        "--disaggregation-transfer-backend",
        "ascend",
        "--watchdog-timeout",
        9000,
        "--context-length",
        "8192",
        "--dtype",
        "bfloat16",
    ]
}


class TestQwen3_480B(TestMultiMixUtils):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 80
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    ttft = 10000
    tpot = 50
    # T: 143@50ms.   800I: xxxxx.     dev：4637@50.34ms
    output_token_throughput = 4600

    def test_qwen3_480b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
