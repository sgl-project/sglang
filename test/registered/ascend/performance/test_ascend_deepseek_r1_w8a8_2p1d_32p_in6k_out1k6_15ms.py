import unittest

from test_ascend_single_mix_utils import NIC_NAME
from test_ascend_disaggregation_utils import TestAscendMultiNodePdSepTestCaseBase


MODEL_PATH = "/root/.cache/modelscope/hub/models/Howeee/DeepSeek-R1-0528-w8a8"

MODEL_CONFIG = {
    "model_path": MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        # "ENABLE_MOE_NZ": "1",
        # "SGLANG_USE_AG_AFTER_QLORA": "1",
        "HCCL_BUFFSIZE": "1536",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        # "ENABLE_MOE_NZ": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "HCCL_BUFFSIZE": "650",
        "SGLANG_DP_ROUND_ROBIN": "1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "4",
        "TASK_QUEUE_ENABLE": "0",
        "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "router_envs": {
        "SGLANG_DP_ROUND_ROBIN": "1",
    },
    "prefill_args": [
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--tp-size",
        16,
        "--mem-fraction-static",
        0.81,
        "--quantization",
        "modelslim",
        "--max-running-requests",
        4,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--max-prefill-tokens",
        28680,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        1,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        2,
        "--dp-size",
        2,
        "--enable-dp-attention",
        "--disable-shared-experts-fusion",
        "--dtype",
        "bfloat16",
        "--enable-attn-tp-input-scattered",
    ],
    "decode_args": [
        "--nnodes",
        "2",
        "--disaggregation-mode",
        "decode",
        "--tp-size",
        32,
        "--dp-size",
        8,
        "--mem-fraction-static",
        0.75,
        "--max-running-requests",
        32,
        "--quantization",
        "modelslim",
        "--moe-a2a-backend",
        "deepep",
        "--enable-dp-attention",
        "--deepep-mode",
        "low_latency",
        "--enable-dp-lm-head",
        "--moe-dense-tp",
        "1",
        "--cuda-graph-bs",
        4,
        "--watchdog-timeout",
        9000,
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        3,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        4,
        "--tokenizer-worker-num",
        4,
        "--prefill-round-robin-balance",
        "--disable-shared-experts-fusion",
        "--dtype",
        "bfloat16",
        "--load-balance-method",
        "decode_round_robin",
    ],
    "router_args": [
        "--mini-lb",
    ],
}

class Test_DeepSeek_R1_W8A8_2P1D_In6000_Out1600(TestAscendMultiNodePdSepTestCaseBase):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    request_rate = 16
    max_concurrency = 32
    num_prompts = 32
    input_len = 6000
    output_len = 1600
    random_range_ratio = 1
    tpot = 20.5
    # T: None   800I A3: None     Dev-800I: 1181/32@19.81
    output_token_throughput = 1058

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
