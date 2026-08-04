import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=600, stage="extra-b", runner_config="deepep-8-gpu-h200")

MODEL_PATH = "/home/t4/models/deepseek-v4-flash-fp8/sgl-project/DeepSeek-V4-Flash-FP8/"
SERVER_LAUNCH_TIMEOUT = 3600


class TestDSV4CPV2Interleave(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "8",
                "--enable-prefill-cp",
                "--cp-strategy",
                "interleave",
                "--attn-cp-size",
                "8",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--mem-fraction-static",
                "0.85",
                "--cuda-graph-max-bs-decode",
                "32",
                "--max-running-requests",
                "32",
                "--watchdog-timeout",
                "900",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
            ],
            env={
                "SGLANG_ENABLE_CP_V2": "1",
                "SGLANG_DSV4_FP4_EXPERTS": "0",
                "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=500,
            num_threads=32,
            num_shots=20,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f'### test_a_gsm8k (dsv4-cp-v2-interleave)\n{metrics["score"]=:.3f}\n'
            )
        self.assertGreater(metrics["score"], 0.935)


if __name__ == "__main__":
    unittest.main()
