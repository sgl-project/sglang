"""GLM-5.3 RoPE DSA DCP acceptance on four Blackwell GPUs."""

import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=1200, stage="extra-b", runner_config="4-gpu-b200")


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.device_count() >= 4
    and all(
        torch.cuda.get_device_capability(i) in ((10, 0), (10, 3)) for i in range(4)
    ),
    "GLM-5.3 DCP requires four SM100/SM103 GPUs",
)
class TestGlm53DCP2(CustomTestCase):
    model = "nvidia/GLM-5.3-NVFP4"
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_score_threshold = 0.95
    gsm8k_num_examples = 200
    gsm8k_num_threads = 32
    gsm8k_accept_length_thres = None
    speculative_args = []

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=[
                "--tp-size",
                "4",
                "--dcp-size",
                "2",
                "--quantization",
                "modelopt_fp4",
                "--reasoning-parser",
                "glm45",
                "--random-seed",
                "0",
                *cls.speculative_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache", timeout=30).raise_for_status()
        # Use the checkpoint's chat template and normal reasoning path.
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                num_examples=self.gsm8k_num_examples,
                num_threads=self.gsm8k_num_threads,
                num_shots=5,
                api="chat",
                temperature=0,
                max_tokens=2048,
            )
        )
        if self.gsm8k_accept_length_thres is not None:
            response = requests.get(self.base_url + "/server_info", timeout=30)
            response.raise_for_status()
            accept_length = response.json()["internal_states"][0][
                "avg_spec_accept_length"
            ]
            print(f"avg_spec_accept_length={accept_length:.4f}")
            self.assertGreater(accept_length, self.gsm8k_accept_length_thres)
        self.assertGreaterEqual(metrics["score"], self.gsm8k_score_threshold)


class TestGlm53DCP2Eagle(TestGlm53DCP2):
    """Wrong draft ownership or target-only collectives degrade acceptance."""

    gsm8k_accept_length_thres = 2.0
    speculative_args = [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
    ]


if __name__ == "__main__":
    unittest.main()
