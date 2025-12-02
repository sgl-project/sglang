import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    QWEN3_8B_MODEL_PATH,
    QWEN3_8B_OTHER_ARGS,
    QWEN3_8B_ENVS,
)


class TestQwen3_8B(TestSingleMixUtils):
    model = QWEN3_8B_MODEL_PATH
    other_args = QWEN3_8B_OTHER_ARGS
    envs = QWEN3_8B_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 16
    input_len = 1000
    output_len = 300
    random_range_ratio = 0.5
    ttft = 291.95
    tpot = 18.83
    output_token_throughput = 404.29

    def test_qwen3_8b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
