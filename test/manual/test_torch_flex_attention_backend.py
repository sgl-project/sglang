"""
Usage:
python3 -m unittest test_torch_flex_attention_backend.TestTorchFlexAttnBackend.test_gsm8k
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestTorchFlexAttnBackend(CustomTestCase):
    def test_gsm8k(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "flex_attention"],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=100,
                num_threads=10,
                num_shots=8,
            )
            metrics = run_eval(args)
            print(f"{metrics=}")
            self.assertGreater(metrics["score"], 0.62)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
