import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=800, suite="nightly-4-gpu-b200", nightly=True)


class FlashinferNvFp4OnlineMoeBackendBase:
    backend = None
    extra_env = {}

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, **cls.extra_env, "SGLANG_ENABLE_JIT_DEEPGEMM": "False"},
            other_args=[
                "--attention-backend",
                "triton",
                "--moe-runner-backend",
                cls.backend,
                "--cuda-graph-max-bs-decode",
                "128",
                "--tp-size",
                "4",
                "--ep-size",
                "4",
                "--quantization",
                "nvfp4_online",
                "--mem-fraction-static",
                "0.7",
            ],
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
        self.assertGreater(metrics["score"], 0.90)


class TestFlashinferTrtllmGenMoeBackendNvFp4Online(
    FlashinferNvFp4OnlineMoeBackendBase, CustomTestCase
):
    backend = "flashinfer_trtllm"
    extra_env = {
        "FLASHINFER_NVFP4_4OVER6": "1",
        "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
        "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
        "SGLANG_FP4_IGNORED_LAYERS": ",".join(
            ["shared_expert"]
            + [f"model.layers.{layer_id}" for layer_id in range(40, 48)]
        ),
    }


class TestFlashinferCuteDSLMoeBackendNvFp4Online(
    FlashinferNvFp4OnlineMoeBackendBase, CustomTestCase
):
    backend = "flashinfer_cutedsl"
    extra_env = {
        "FLASHINFER_NVFP4_4OVER6": "1",
        "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
        "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
    }


if __name__ == "__main__":
    unittest.main()
