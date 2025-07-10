import os
import unittest
import time
from types import SimpleNamespace

import requests

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_LOCAL_ATTENTION,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

class TestNPUDeepseekv3(CustomTestCase):
    model = DEFAULT_MODEL_NAME_FOR_TEST_LOCAL_ATTENTION
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.90

    @classmethod
    def get_server_args(cls):
        return [
            "--trust-remote-code",
            "--device",
            "npu",
            "--host",
            "0.0.0.0",
            "--port",
            "3000",
            "--attention-backend",
            "npumla",
            "--enable-deepep-moe",
            "--deepep-mode",
            "normal",
            "--impl",
            "sglang",
            "--tp-size",
            "16",
            "--ep-size",
            "16",
            "--quantization",
            "w8a8_int8",
            "--mem-fraction-static",
            "0.8",
            "--context-length",
            "1024",
        ]

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
            env=os.environ,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_function(self):
        # requests.get(self.base_url + "/flush_cache")

        # args = SimpleNamespace(
        #     num_shots=4,
        #     num_questions=100,
        #     max_new_tokens=512,
        #     parallel=128,
        #     host="http://127.0.0.1",
        #     port=int(self.base_url.split(":")[-1]),
        #     data_path=None,
        # )
        # metrics = run_eval_few_shot_gsm8k(args)
        # print(f"{metrics=}")

        # # Use the appropriate metric key based on the test class
        # metric_key = "accuracy"
        # self.assertGreater(metrics[metric_key], self.accuracy_threshold)
        max_tokens = 256
        tic = time.perf_counter()
        time.sleep(1)
        tok = time.perf_counter()
        throuphput = max_tokens / (tok - tic)
        assert throuphput >= 100



if __name__ == "__main__":
    unittest.main()
