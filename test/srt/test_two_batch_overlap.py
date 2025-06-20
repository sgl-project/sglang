import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.two_batch_overlap import compute_split_seq_index
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST,
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestTwoBatchOverlap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--enable-deepep-moe",
                "--deepep-mode",
                "normal",
                "--disable-cuda-graph",  # DeepEP normal does not support CUDA Graph
                "--enable-two-batch-overlap",
            ],
            env={"SGL_ENABLE_JIT_DEEPGEMM": "0", **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_generate_single_prompt(self):
        response = requests.post(
            self.base_url + "/generate",
            # we use an uncommon start to minimise the chance that the cache is hit by chance
            json={
                "text": "_ 1+1=2, 1+2=3, 1+3=4, 1+4=",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
        )
        print(f"{response.json()=}")
        self.assertEqual(response.json()["text"], "5, 1+5=6")

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


class TestTwoBatchOverlapUnitTest(unittest.TestCase):
    # TODO change tests when having 6328
    def test_compute_split_seq_index(self):
        for num_tokens, expect in [
            (0, 0),
            (100, 50),
            (99, 49),
        ]:
            actual = compute_split_seq_index(
                forward_mode=ForwardMode.DECODE, num_tokens=num_tokens, extend_lens=None
            )
            self.assertEqual(actual, expect)

        for extend_lens, expect in [
            ([], 0),
            ([42], 0),
            ([42, 999], 1),
            ([999, 42], 1),
            ([4096, 4096, 4096, 4096], 2),
            ([4095, 4096, 4096, 4096, 1], 2),
            ([1, 4095, 4096, 4096, 4096], 3),
            ([4097, 4096, 4096, 4095, 1], 2),
            ([1, 1, 1, 1, 99999], 4),
            ([99999, 1, 1, 1, 1], 1),
        ]:
            actual = compute_split_seq_index(
                forward_mode=ForwardMode.EXTEND,
                num_tokens=None,
                extend_lens=extend_lens,
            )
            print(f"{extend_lens=} {expect=} {actual=}")
            self.assertEqual(actual, expect)


class TestQwen3TwoBatchOverlap(TestTwoBatchOverlap):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--enable-deepep-moe",
                "--deepep-mode",
                "normal",
                "--disable-cuda-graph",  # DeepEP normal does not support CUDA Graph
                "--enable-two-batch-overlap",
            ],
            env={"SGL_ENABLE_JIT_DEEPGEMM": "0", **os.environ},
        )


if __name__ == "__main__":
    unittest.main()
