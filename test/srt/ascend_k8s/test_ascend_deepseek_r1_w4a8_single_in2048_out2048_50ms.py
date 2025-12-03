import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
)

# DEEPSEEK_R1_0528_W4A8_MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-R1-0528-w4a8"
MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Howeee/DeepSeek-R1-0528-w8a8"
MODEL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "8",
    "HCCL_BUFFSIZE": "1300",
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
        "w8a8_int8",
        "--watchdog-timeout",
        "9000",
        "--cuda-graph-bs",
        "8",
        "16",
        # "24",
        # "28",
        # "32",
        "--mem-fraction-static",
        "0.8",
        "--max-running-requests",
        "16",
        "--context-length",
        "8188",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "-1",
        "--max-prefill-tokens",
        "6000",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--enable-dp-attention",
        "--dp-size",
        "4",
        "--enable-dp-lm-head",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "1",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "2",
        "--dtype",
        "bfloat16",
    ]
)


class Test_Ascend_DeepSeek_R1_W4A8_In2048_Out2048(TestSingleMixUtils):
    model = MODEL_PATH
    other_args = MODEL_OTHER_ARGS
    envs = MODEL_ENVS
    dataset_name = "random"
    request_rate = 16
    max_concurrency = 4
    num_prompts = int(max_concurrency) * 4
    input_len = 2048
    output_len = 2048
    random_range_ratio = 1
    ttft = 5318.34
    tpot = 46.66
    output_token_throughput = 3552.66

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
