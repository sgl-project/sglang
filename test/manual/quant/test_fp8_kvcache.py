import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestFp8KvcacheBase(CustomTestCase):
    model_config = None

    @classmethod
    def setUpClass(cls):
        if cls.model_config is None:
            raise NotImplementedError("model_config must be specified in subclass")

        cls.model = cls.model_config["model_name"]
        cls.base_url = DEFAULT_URL_FOR_TEST
        dirpath = os.path.dirname(__file__)
        config_file = os.path.join(dirpath, cls.model_config["config_filename"])

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--quantization-param-path",
                config_file,
            ],
        )


class TestFp8KvcacheLlama(TestFp8KvcacheBase):
    model_config = {
        "model_name": DEFAULT_MODEL_NAME_FOR_TEST,
        "config_filename": "kv_cache_scales_llama3_8b.json",
    }

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.80)

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


class TestFp8KvcacheQwen(TestFp8KvcacheBase):
    model_config = {
        "model_name": DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
        "config_filename": "kv_cache_scales_qwen2_1_5b.json",
    }

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.01)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.3)


if __name__ == "__main__":
    unittest.main()
