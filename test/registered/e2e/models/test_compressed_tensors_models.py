# Model tests for compressed tensors (FP8)

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=63, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=42, suite="stage-b-test-1-gpu-small-amd")


class TestCompressedTensorsLlama3FP8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "RedHatAI/Meta-Llama-3.1-8B-FP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[],
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
        if is_hip():
            # Lower threshold for AMD because FP8 dtype differs (fp8_fnuz)
            self.assertGreaterEqual(metrics["score"], 0.40)
        else:
            self.assertGreaterEqual(metrics["score"], 0.45)


@unittest.skipIf(is_hip(), "This loading regression targets the CUDA backend")
class TestStaticInt8CheckpointLoading(CustomTestCase):
    def test_static_checkpoint_loads_and_decodes(self):
        # CUDA currently quantizes activations dynamically; this checks that a
        # static compressed-tensors checkpoint loads and decodes on CUDA.
        model = "nm-testing/Qwen2.5-0.5B-W8A8_tensor_weight_static_per_tensor_act-e2e"
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[],
        )
        try:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                },
                timeout=60,
            )
            response.raise_for_status()
            self.assertTrue(response.json()["text"])
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
