import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    Qwen3_Next_80B_A3B_MODEL_PATH,
    Qwen3_Next_80B_A3B_OTHER_ARGS,
    Qwen3_Next_80B_A3B_ENVS,
)


class TestQwen3_Next_80B_A3B(TestSingleMixUtils):
    model = Qwen3_Next_80B_A3B_MODEL_PATH
    other_args = Qwen3_Next_80B_A3B_OTHER_ARGS
    envs = Qwen3_Next_80B_A3B_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 32
    input_len = 1000
    output_len = 300
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 100
    output_token_throughput = 300

    def test_qwen3_next_80b_a3b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
