"""
Usage:
python3 -m unittest test_torch_flex_attention_backend.TestTorchFlexAttnBackend.test_gsm8k
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
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
                num_shots=8,
                data_path=None,
                num_questions=100,
                parallel=10,
                max_new_tokens=512,
                host="http://127.0.0.1",
                port=int(base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            print(f"{metrics=}")
            self.assertGreater(metrics["accuracy"], 0.62)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
