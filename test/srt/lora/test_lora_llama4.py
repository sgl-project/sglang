import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODELS = [
    SimpleNamespace(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        tp_size=8,
    ),
]


class TestLlama4LoRA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_bringup(self):
        for model in MODELS:
            try:
                process = popen_launch_server(
                    model.model,
                    self.base_url,
                    timeout=3 * DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        "--enable-lora",
                        "--max-lora-rank",
                        "64",
                        "--lora-target-modules",
                        "all",
                        "--tp-size",
                        str(model.tp_size),
                        "--context-length",
                        "262144",
                        "--attention-backend",
                        "fa3",
                    ],
                )
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
