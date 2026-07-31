import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=300, stage="extra-b", runner_config="8-gpu-h200")

MODEL_FP8 = "sgl-project/DeepSeek-V4-Flash-FP8"
GSM8K_ACCURACY_THRESHOLD = 0.93


class TestDeepseekV4FlashIndexCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_FP8
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--moe-runner-backend",
                "marlin",
                "--json-model-override-args",
                '{"index_topk_freq": 4}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
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
                f"### test_gsm8k ({type(self).__name__})\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )

        self.assertGreater(metrics["accuracy"], GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
