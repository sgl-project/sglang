from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(est_time=313, suite="stage-b-test-2-gpu-large-amd")


class TestMixtralAccuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--tp",
            "2",
            "--mem-fraction-static",
            "0.9",
            "--context-length",
            "38768",
            "--quantization",
            "quark_int4fp8_moe",
            "--attention-backend",
            "triton",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=45 * 60,
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
            num_examples=1400,
            num_threads=128,
            num_shots=8,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.56)


if __name__ == "__main__":
    import unittest

    unittest.main()
