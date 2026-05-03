import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

# FlashAttention4 integration test (requires SM 100+ / Blackwell B200)
register_cuda_ci(est_time=265, suite="stage-b-test-4-gpu-b200")


@unittest.skipIf(get_device_sm() < 100, "Test requires CUDA SM 100 or higher")
class TestFlashAttention4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--attention-backend",
            "fa4",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1319,
            num_threads=200,
        )
        metrics = run_eval(args)
        print(metrics)

        self.assertGreater(metrics["score"], 0.89)


@unittest.skipIf(get_device_sm() < 100, "Test requires CUDA SM 100 or higher")
class TestFlashAttention4SpeculativeDecodeTopk(unittest.TestCase):
    """Test FlashAttention4 with EAGLE3 speculative decoding (topk > 1).

    Verifies that FA4 + EAGLE3 topk > 1 produces correct outputs and
    achieves meaningful speculative acceptance length.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--attention-backend",
            "fa4",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex",
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            "4",
            "--speculative-num-draft-tokens",
            "8",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1319,
            num_threads=200,
        )
        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["score"], 0.89)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 1.5)


if __name__ == "__main__":
    unittest.main()
