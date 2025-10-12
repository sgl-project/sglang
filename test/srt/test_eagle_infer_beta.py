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


class TestEagleBS1(CustomTestCase):
    num_questions = 60

    @classmethod
    def setUpClass(cls):
        """
        Prepare the test class by setting model and base URL and launching a server process configured for EAGLE speculative inference; the launched process is stored on cls.process.
        """
        cls.model = "meta-llama/Llama-2-7b-chat-hf"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                "--enable-beta-spec",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model",
                "lmzheng/sglang-EAGLE-llama2-chat-7B",
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "6",
                "--max-running-requests",
                "1",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        """
        Terminate the server process started in setUpClass, including any child processes.
        """
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        """
        Run GSM8K-style evaluation against the launched model server and assert the accuracy meets the expected threshold.
        
        Constructs evaluation arguments (5-shot, specified number of questions, max_new_tokens=512, parallel=128) using the test's base URL, executes run_eval, prints the resulting metrics, and asserts that `metrics["accuracy"]` is greater than 0.33.
        """
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=self.num_questions,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"TestEagleBS1 -- {metrics=}")
        self.assertGreater(
            metrics["accuracy"], 0.33
        )  # 0.3333 for 60 questions; 0.234 for 1319 questions


class TestEagleLargeBS(CustomTestCase):
    num_questions = 10000
    max_running_requests = 64
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "triton",
        "--enable-beta-spec",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model",
        "lmzheng/sglang-EAGLE-llama2-chat-7B",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
        "--mem-fraction-static",
        "0.75",
        "--max-running-requests",
        str(max_running_requests),
        "--cuda-graph-bs",
        *[str(i) for i in range(1, max_running_requests + 1)],
    ]

    @classmethod
    def setUpClass(cls):
        """
        Prepare the test class by starting the model server and recording connection and process details.
        
        Sets the class attributes `model` and `base_url`, launches the model server using the class configuration, and stores the resulting server process on `cls.process` for later teardown and use by tests.
        """
        cls.model = "meta-llama/Llama-2-7b-chat-hf"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        """
        Terminate the server process started in setUpClass, including any child processes.
        """
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        """
        Run a GSM8K-style evaluation against the launched model server and assert the model meets the accuracy threshold.
        
        Constructs evaluation arguments (5 shots, max_new_tokens=512, parallel=128, questions taken from self.num_questions, host and port derived from self.base_url), runs run_eval, prints the resulting metrics, and asserts that metrics["accuracy"] > 0.23.
        """
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=self.num_questions,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"TestEagleLargeBS -- {metrics=}")
        self.assertGreater(
            metrics["accuracy"], 0.23
        )  # 0.3333 for 60 questions; 0.234 for 1319 questions


if __name__ == "__main__":
    unittest.main()