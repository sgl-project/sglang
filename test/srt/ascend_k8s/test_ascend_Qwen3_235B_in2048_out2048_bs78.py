import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    QWEN3_235B_MODEL_PATH,
    QWEN3_235B_OTHER_ARGS,
    QWEN3_235B_ENVS,
)


class TestQwen3_235B(TestSingleMixUtils):
    model = QWEN3_235B_MODEL_PATH
    other_args = QWEN3_235B_OTHER_ARGS
    envs = QWEN3_235B_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 78
    input_len = 2048
    output_len = 2048
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 100
    output_token_throughput = 300

    def test_qwen3_235b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
