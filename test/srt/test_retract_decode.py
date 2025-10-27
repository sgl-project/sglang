import time
import unittest
from types import SimpleNamespace

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


class TestRetractDecode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_TEST_RETRACT.override(True):
            cls.process = popen_launch_server(
                cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
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


class TestRetractDecodeChunkCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_TEST_RETRACT.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--disable-radix-cache", "--chunked-prefill-size", 128],
            )

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

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
