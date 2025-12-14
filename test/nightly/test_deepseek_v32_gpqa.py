import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    try_cached_model,
    write_github_step_summary,
)

register_cuda_ci(est_time=3600, suite="nightly-8-gpu-b200", nightly=True)

# Use the latest version of DeepSeek-V3.2
DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"
SERVER_LAUNCH_TIMEOUT = 1200


class TestDeepseekV32Accuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(DEEPSEEK_V32_MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--enable-dp-attention",
            "--dp",
            "8",
            "--tool-call-parser",
            "deepseekv32",
            "--reasoning-parser",
            "deepseek-v3",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true,"num_threads": 64}',
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
    ):
        args = SimpleNamespace(
            num_shots=20,
            data_path=None,
            num_questions=1400,
            parallel=1400,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")
        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v32)\n" f'{metrics["accuracy"]=:.3f}\n'
            )
        self.assertGreater(metrics["accuracy"], 0.935)

    def test_gpqa(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=DEEPSEEK_V32_MODEL_PATH,
            eval_name="gpqa",
            num_examples=198,
            # use enough threads to allow parallelism
            num_threads=198,
            max_tokens=120000,
            thinking_mode="deepseek-v3",
            temperature=0.1,
            # Repeat 4 times for shorter runtime. Ideally we should repeat at least 8 times.
            repeat=4,
        )

        print(f"Evaluation start for gpqa")
        metrics = run_eval(args)
        print(f"Evaluation end for gpqa: {metrics=}, expected_score=0.835")
        mean_score = metrics["mean_score"]
        self.assertGreaterEqual(mean_score, 0.835)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gpqa (deepseek-v32)\n" f"Mean Score: {mean_score:.3f}\n"
            )


if __name__ == "__main__":
    unittest.main()
