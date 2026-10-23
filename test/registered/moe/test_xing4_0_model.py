import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="2-gpu-large")


class TestXing4_0(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "XingChen-AGI/Xing4.0-29B-A4B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "2",
                "--context-length",
                "8192",
                "--trust-remote-code",
                "--reasoning-parser",
                "xing4_0",
                "--tool-call-parser",
                "xing4_0",
            ],
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
            num_examples=100,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        # Conservative floor: catches load/correctness regressions (broken
        # weights, misplaced reductions); tighten once a reference score
        # is established.
        self.assertGreater(metrics["score"], 0.3)


if __name__ == "__main__":
    unittest.main()
