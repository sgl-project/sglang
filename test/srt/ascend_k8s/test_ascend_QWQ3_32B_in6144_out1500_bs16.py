import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    QWQ_32B_MODEL_PATH,
    QWQ_32B_OTHER_ARGS,
    QWQ_32B_ENVS,
)


class TestQWQ_32B(TestSingleMixUtils):
    model = QWQ_32B_MODEL_PATH
    other_args = QWQ_32B_OTHER_ARGS
    envs = QWQ_32B_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 16
    input_len = 6144
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 22.79
    output_token_throughput = 300

    def test_qwq_32b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
