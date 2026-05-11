import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_is_hip = is_hip()

register_cuda_ci(est_time=90, suite="stage-b-test-large-2-gpu")
register_amd_ci(est_time=90, suite="stage-b-test-large-2-gpu-amd")


class TestKimiLinear(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--tp-size", "2", "--trust-remote"]
        if _is_hip:
            # AMD/ROCm: the aiter absorbed-MLA decode kernel is
            # numerically lossy for Kimi-Linear (skip-rope MLA + KDA hybrid),
            # so route decode through triton while keeping aiter for prefill.
            # Same pattern as `test/registered/amd/test_kimi_k2_instruct.py`.
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
        self.assertGreater(metrics["accuracy"], 0.88)


if __name__ == "__main__":
    unittest.main()
