import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as gsm8k_run_eval
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestInt8vcacheBase(unittest.TestCase):
    model_config = None

    @classmethod
    def setUpClass(cls):
        if cls.model_config is None:
            raise NotImplementedError("model_config must be specified in subclass")

        cls.model = cls.model_config["model_name"]
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--kv-cache-dtype",
                "int8",
            ],
        )


class TestInt8KvcacheLlamaGQA(TestInt8vcacheBase):
    model_config = {
        "model_name": DEFAULT_MODEL_NAME_FOR_TEST,
    }

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
        self.assertGreaterEqual(metrics["score"], 0.70)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = gsm8k_run_eval(args)
        self.assertGreater(metrics["accuracy"], 0.74)


class TestInt8KvcacheLlamaMHA(TestInt8vcacheBase):
    model_config = {
        "model_name": DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    }

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = gsm8k_run_eval(args)
        # base 0.25
        self.assertGreater(metrics["accuracy"], 0.24)


if __name__ == "__main__":
    unittest.main()
