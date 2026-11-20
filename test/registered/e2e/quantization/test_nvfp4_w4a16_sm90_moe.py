"""--moe-runner-backend nvfp4_w4a16_sm90 on a real ModelOpt NVFP4 MoE checkpoint: GSM8K through the server, with
CUDA graphs. Reload under captured graphs is covered by test_modelopt_nvfp4_sm90_runner.py, because a full
update_weights_from_disk of this checkpoint fails before the MoE layers on every backend (a dense NVFP4 linear parameter
has no weight_loader after loading)."""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-large")


class TestNvfp4W4a16Sm90Moe(CustomTestCase):
    model = "nvidia/Qwen3-30B-A3B-NVFP4"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--moe-runner-backend",
                "nvfp4_w4a16_sm90",
                "--mem-fraction-static",
                "0.8",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        # With thinking off, Marlin (the default on Hopper) scores 0.948 and this backend 0.938 on these 500 questions
        # on an H100. A broken kernel scores near 0, and 0.90 leaves room for sampling noise.
        metrics = run_sgl_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                num_examples=500,
                num_threads=128,
                chat_template_kwargs={"enable_thinking": False},
            )
        )
        self.assertGreaterEqual(metrics["score"], 0.90)


if __name__ == "__main__":
    unittest.main()
