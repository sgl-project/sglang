import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestDoubleSparsity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        dirpath = os.path.dirname(__file__)
        config_file = os.path.join(dirpath, "Llama-3.1-8B-Instruct.json")
        # NOTE: Generate the config file by running https://github.com/andy-yang-1/DoubleSparse/blob/main/evaluation/group_channel_config.py
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-double-sparsity",
                "--ds-channel-config-path",
                config_file,
                "--ds-heavy-channel-num",
                "32",
                "--ds-heavy-channel-type",
                "k",
                "--ds-heavy-token-num",
                "512",
                "--ds-sparse-decode-threshold",
                "0",
                "--max-total-tokens",
                "200000",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid, include_self=True)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        assert metrics["score"] >= 0.65


if __name__ == "__main__":
    unittest.main()
