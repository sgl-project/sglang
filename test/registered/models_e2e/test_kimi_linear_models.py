import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=178, suite="stage-b-test-2-gpu-large")
register_amd_ci(est_time=178, suite="stage-b-test-2-gpu-large-amd")


class TestKimiLinear(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--tp-size", "2", "--trust-remote"]
        if is_hip():
            # AMD/ROCm: the aiter absorbed-MLA decode kernel is numerically
            # lossy for Kimi-Linear (skip-rope MLA + KDA hybrid), so route
            # decode through triton while keeping aiter for prefill. Same
            # pattern as `test/registered/amd/test_kimi_k2_instruct.py`.
            other_args += [
                "--decode-attention-backend",
                "triton",
                "--prefill-attention-backend",
                "aiter",
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
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.88)


if __name__ == "__main__":
    unittest.main()
