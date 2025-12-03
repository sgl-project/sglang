import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import is_in_ci


class TestRetractDecode(CustomTestCase):
    """python -m unittest test_retract_decode.TestRetractDecode"""

    other_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = ["--chunked-prefill-size", "128"] + cls.other_args
        with envs.SGLANG_TEST_RETRACT.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=launch_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)
        time.sleep(1)  # wait for mem check

        assert self.process.poll() is None, "Server crashed during test"


class TestRetractDecodePaged(TestRetractDecode):
    """python -m unittest test_retract_decode.TestRetractDecodePaged"""

    other_args = ["--page-size", "16"]


class TestRetractDecodeChunkCache(TestRetractDecode):
    """python -m unittest test_retract_decode.TestRetractDecodeChunkCache"""

    other_args = ["--disable-radix-cache"]


class TestRetractDecodeChunkCachePaged(TestRetractDecode):
    """python -m unittest test_retract_decode.TestRetractDecodeChunkCachePaged"""

    other_args = ["--disable-radix-cache", "--page-size", "16"]


@unittest.skipIf(is_in_ci(), "Skipped in CI due to long runtime")
class TestRetractDecodeLongOutput(CustomTestCase):
    """python -m unittest test_retract_decode.TestRetractDecodeLongOutput"""

    other_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--chunked-prefill-size",
            "128",
            "--page-size",
            "16",
        ] + cls.other_args
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )

    def test_long_output_retract(self):
        data = {
            "input_ids": [[233 + i] * 1234 for i in range(256)],
            "sampling_params": {"max_new_tokens": 90000, "ignore_eos": True},
        }
        res = requests.post(f"{self.base_url}/generate", json=data)
        assert res.status_code == 200, f"Request failed: {res.status_code}"
        assert self.process.poll() is None, "Server crashed during test"


@unittest.skipIf(is_in_ci(), "Skipped in CI due to long runtime")
class TestRetractDecodeLongOutputChunkCache(TestRetractDecodeLongOutput):
    """python -m unittest test_retract_decode.TestRetractDecodeLongOutputChunkCache"""

    other_args = ["--disable-radix-cache"]


if __name__ == "__main__":
    unittest.main()
