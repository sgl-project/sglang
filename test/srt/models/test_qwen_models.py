import random
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

STANDARD_MODELS = [
    SimpleNamespace(model="Qwen/Qwen3-4B", accuracy=0.85),
    SimpleNamespace(
        model="Qwen/Qwen3-14B",
        accuracy=0.87,
    ),
    SimpleNamespace(model="Qwen/Qwen3-32B", accuracy=0.87),
]

FP8_MODELS = [
    SimpleNamespace(model="Qwen/Qwen3-4B-FP8", accuracy=0.85),
    SimpleNamespace(
        model="Qwen/Qwen3-14B-FP8",
        accuracy=0.85,
    ),
    SimpleNamespace(model="Qwen/Qwen3-32B-FP8", accuracy=0.85),
]


class TestQwen3(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_gsm8k(self):

        models_to_test = STANDARD_MODELS + FP8_MODELS

        if is_in_ci():
            models_to_test = [random.choice(STANDARD_MODELS)] + [
                random.choice(FP8_MODELS)
            ]

        for model in models_to_test:
            try:
                process = popen_launch_server(
                    model.model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=["--reasoning-parser", "qwen3"],
                )
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
                self.assertGreaterEqual(metrics["accuracy"], model.accuracy)
            except Exception as e:
                print(f"Error testing {model.model}: {e}")
                self.fail(f"Test failed for {model.model}: {e}")

            finally:
                # Ensure process cleanup happens regardless of success/failure
                if process is not None and process.poll() is None:
                    print(f"Cleaning up process {process.pid}")
                    try:
                        kill_process_tree(process.pid)
                    except Exception as e:
                        print(f"Error killing process: {e}")


if __name__ == "__main__":
    unittest.main()
