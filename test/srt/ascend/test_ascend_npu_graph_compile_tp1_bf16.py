import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEFAULT_MODEL_NAME_FOR_TEST = "Qwen/Qwen2.5-7B-Instruct"


class TestAscendNpuGraphCompile(CustomTestCase):
    def test_gsm8k(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        url = urlparse(base_url)
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.7,
                "--enable-torch-compile",
                "--watchdog-timeout",
                30000,
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=1319,
                max_new_tokens=512,
                parallel=128,
                host=f"http://{url.hostname}",
                port=int(url.port),
            )

            metrics = run_eval_few_shot_gsm8k(args)
            self.assertGreaterEqual(metrics["accuracy"], 0.62)
            self.assertLessEqual(metrics["latency"], 150)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
