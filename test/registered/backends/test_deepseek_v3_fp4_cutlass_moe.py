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

register_cuda_ci(est_time=900, suite="nightly-4-gpu-b200", nightly=True)

FULL_DEEPSEEK_V3_FP4_MODEL_PATH = "nvidia/DeepSeek-V3-0324-FP4"
SERVER_LAUNCH_TIMEOUT = 1000


class TestDeepseekV3FP4CutlassMoE(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FULL_DEEPSEEK_V3_FP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "4",
            "--ep",
            "4",
            "--attention-backend",
            "trtllm_mla",
            "--moe-runner-backend",
            "flashinfer_cutlass",
            "--quantization",
            "modelopt_fp4",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1319,
            num_threads=1319,
            num_shots=8,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v3-fp4-cutlass-moe)\n"
                f'{metrics["score"]=:.3f}\n'
            )
            self.assertGreater(metrics["score"], 0.935)


if __name__ == "__main__":
    unittest.main()
