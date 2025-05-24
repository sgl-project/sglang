import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
)


class _BaseTestTwoBatchOverlap(unittest.TestCase):
    extra_args = None
    skip_in_ci = False

    @classmethod
    def setUpClass(cls):
        if is_in_ci() and cls.skip_in_ci:
            return

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
                "--enable-two-batch-overlap",
                *cls.extra_args,
            ],
            env={"SGL_ENABLE_JIT_DEEPGEMM": "0", **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        if is_in_ci() and cls.skip_in_ci:
            return

        kill_process_tree(cls.process.pid)

    def test_generate_single_prompt(self):
        if is_in_ci() and self.skip_in_ci:
            return

        response = requests.post(
            self.base_url + "/generate",
            # we use an uncommon start to minimise the chance that the cache is hit by chance
            json={
                "text": "_ 1+1=2, 1+2=3, 1+3=4, 1+4=",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
        )
        print(f"{response.json()=}")
        self.assertEquals(response.json()["text"], "5, 1+5=6")

    def test_mmlu(self):
        if is_in_ci() and self.skip_in_ci:
            return

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


class TestTwoBatchOverlapNormal(_BaseTestTwoBatchOverlap):
    extra_args = [
        "--deepep-mode",
        "normal",
        # DeepEP normal does not support CUDA Graph
        "--disable-cuda-graph",
    ]


class TestTwoBatchOverlapLowLatency(_BaseTestTwoBatchOverlap):
    extra_args = [
        "--deepep-mode",
        "low_latency",
        "--chunked-prefill-size",
        "128",
    ]
    # DeepEP low-latency requires IB
    skip_in_ci = True


if __name__ == "__main__":
    unittest.main()
