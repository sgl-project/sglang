import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_blackwell, kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestNvidiaNemotronNanoV2(CustomTestCase):
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    accuracy = 0.87

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--max-mamba-cache-size",
                "256",
            ],
        )

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
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["accuracy"], self.accuracy)


class TestNvidiaNemotronNanoV2FP8(TestNvidiaNemotronNanoV2):
    accuracy = 0.87
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8"


@unittest.skipIf(not is_blackwell(), "NVFP4 only supported on blackwell")
class TestNvidiaNemotronNanoV2NVFP4(TestNvidiaNemotronNanoV2):
    accuracy = 0.855
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4"


if __name__ == "__main__":
    unittest.main()
