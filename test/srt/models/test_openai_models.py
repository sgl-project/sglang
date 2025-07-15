import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestOpenAIMoE(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "path-to/Orangina"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--cuda-graph-max-bs",
                "128",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # def test_gsm8k(self):
    #     args = SimpleNamespace(
    #         num_shots=5,
    #         data_path=None,
    #         num_questions=200,
    #         max_new_tokens=512,
    #         parallel=128,
    #         host="http://127.0.0.1",
    #         port=int(self.base_url.split(":")[-1]),
    #     )
    #     metrics = run_eval_few_shot_gsm8k(args)
    #     print(f"Eval accuracy of GSM8K: {metrics=}")
    #     # self.assertGreater(metrics["accuracy"], 0.71) # target

    # def test_mmlu(self):
    #     args = SimpleNamespace(
    #         base_url=self.base_url,
    #         model=self.model,
    #         eval_name="mmlu",
    #         num_examples=64,
    #         num_threads=32,
    #     )

    #     metrics = run_eval(args)
    #     print(f"Eval accuracy of MMLU: {metrics=}")
    #     # self.assertGreaterEqual(metrics["score"], 0.759) # target

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=10, prompt="Human: How are you?\n\nAssistant:") # What is the capital of France?
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")


if __name__ == "__main__":
    unittest.main()
