import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_MODEL_PATH,
    QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_OTHER_ARGS,
    QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_ENVS,
)


class TestQwen3_Coder_480B_A35b_Instruct_W8a8_Quarot(TestSingleMixUtils):
    model = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_MODEL_PATH
    other_args = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_OTHER_ARGS
    envs = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 32
    input_len = 16000
    output_len = 10000
    random_range_ratio = 0.5
    ttft = 1206.81
    tpot = 36.45
    output_token_throughput = 252

    def test_qwen3_coder_480b_a35b_instruct_w8a8_quarot(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
