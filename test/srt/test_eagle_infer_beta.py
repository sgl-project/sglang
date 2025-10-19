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


class TestEagleServerBase(CustomTestCase):
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
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1000,
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
