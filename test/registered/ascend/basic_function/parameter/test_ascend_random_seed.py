import random
import unittest

import requests
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestRandomSeedZero(CustomTestCase):
    """Testcase：Verify set --random-seed parameter, with the same random_seed the model's output is consistent.

       [Test Category] Parameter
       [Test Target] --random-seed
       """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    random_seed = random.randint(0, 1000000)
    request_times=10
    @classmethod
    def setUpClass(cls):
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--random-seed",
            cls.random_seed,
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_random_seed(self):
        response_texts = []

        for i in range(self.request_times):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            response_text = response.json()["text"]
            response_texts.append(response_text)
        first_text = response_texts[0]
        for idx, text in enumerate(response_texts[1:], start=2):
            self.assertEqual(
                text,
                first_text,
                f"different response_text"
            )

if __name__ == "__main__":
    unittest.main()
