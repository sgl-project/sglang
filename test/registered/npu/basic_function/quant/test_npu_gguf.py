import logging
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_4B_GGUF_Q4_K_M_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval as run_gsm8k_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="full-1-npu-a3", nightly=True)

logger = logging.getLogger(__name__)

TEST_MODEL_MATRIX = {
    QWEN3_4B_GGUF_Q4_K_M_WEIGHTS_PATH: {
        "accuracy": 0.80,
    },
}


class TestAscendGGUF(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.common_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
        ]

    def test_a_gsm8k(self):
        for model in self.models:
            with self.subTest(model=model):
                logger.info(f"##=== Testing accuracy: {model} ===##")

                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        *self.common_args,
                    ],
                )

                try:
                    args = SimpleNamespace(
                        eval_name="gsm8k",
                        num_examples=1319,
                        max_tokens=512,
                        num_threads=128,
                        host=f"http://{self.url.hostname}",
                        port=int(self.url.port),
                    )

                    metrics = run_gsm8k_eval(args)
                    self.assertGreaterEqual(
                        metrics["accuracy"],
                        TEST_MODEL_MATRIX[model]["accuracy"],
                    )
                finally:
                    kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
