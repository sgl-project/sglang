import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODELS = [
    SimpleNamespace(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        accuracy=0.9,
        tp_size=4,
    ),
]


class TestLlama4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_gsm8k(self):

        for model in MODELS:
            try:
                process = popen_launch_server(
                    model.model,
                    self.base_url,
                    timeout=3 * DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        "--chat-template",
                        "llama-4",
                        "--tp-size",
                        str(model.tp_size),
                        "--mem-fraction-static",
                        "0.8",
                        "--context-length",
                        "8192",
                    ],
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
