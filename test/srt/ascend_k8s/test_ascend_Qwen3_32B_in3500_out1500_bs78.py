import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    QWEN3_32B_MODEL_PATH,
    QWEN3_32B_OTHER_ARGS,
    QWEN3_32B_ENVS,
)


class TestQwen3_32B(TestSingleMixUtils):
    model = QWEN3_32B_MODEL_PATH
    other_args = QWEN3_32B_OTHER_ARGS
    envs = QWEN3_32B_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 78
    input_len = 3500
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 49.4
    output_token_throughput = 650.07

    def test_qwen3_32b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
