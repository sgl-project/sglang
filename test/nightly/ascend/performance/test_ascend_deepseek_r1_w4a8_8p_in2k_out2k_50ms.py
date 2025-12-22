import unittest

from test_ascend_single_mix_utils import TestSingleMixUtils, NIC_NAME

# DEEPSEEK_R1_0528_W4A8_MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-R1-0528-w4a8"
#MODEL_PATH = "/root/.cache/modelscope/hub/models/Howeee/DeepSeek-R1-0528-w8a8"
MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel"

MODEL_ENVS = {
    # "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "HCCL_BUFFSIZE": "1600",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_USE_FIA_NZ": "1",
    "ENABLE_MOE_NZ": "1",
}
MODEL_OTHER_ARGS = (
    [
        "--tp",
        "16",
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "modelslim",
        "--watchdog-timeout",
        "9000",
        "--cuda-graph-bs",
        "4",
        "8",
        "16",
        "--mem-fraction-static",
        "0.74",
        "--max-running-requests",
        "256",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "-1",
        "--max-prefill-tokens",
        "1500",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--enable-dp-attention",
        "--dp-size",
        "16",
        "--enable-dp-lm-head",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--dtype",
        "bfloat16",
    ]
)


class Test_Ascend_DeepSeek_R1_W4A8_In2048_Out2048(TestSingleMixUtils):
    model = MODEL_PATH
    other_args = MODEL_OTHER_ARGS
    envs = MODEL_ENVS
    dataset_name = "random"
    max_concurrency = 256
    num_prompts = int(max_concurrency) * 4
    input_len = 2048
    output_len = 2048
    random_range_ratio = 1
    ttft = 10000
    tpot = 50
    # T: 143@50ms. 800I A3: 1.8*T
    output_token_throughput = 143 * 1.8 * 8 / 0.93

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
