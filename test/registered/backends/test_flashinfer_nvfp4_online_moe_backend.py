import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=800, stage="nightly", runner_config="4-gpu-b200")


class FlashinferNvFp4OnlineMoeBackendBase:
    backend = None
    model = None
    extra_args = []
    extra_env = {}
    eval_args = {}
    spec_accept_length_threshold = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, **cls.extra_env, "SGLANG_ENABLE_JIT_DEEPGEMM": "False"},
            other_args=[
                *cls.extra_args,
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
            num_examples=200,
            num_threads=128,
            **self.eval_args,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.90)
        if self.spec_accept_length_threshold is not None:
            server_info = requests.get(self.base_url + "/server_info").json()
            avg_spec_accept_length = server_info["internal_states"][0][
                "avg_spec_accept_length"
            ]
            print(f"{avg_spec_accept_length=}")
            self.assertGreater(
                avg_spec_accept_length, self.spec_accept_length_threshold
            )


# Only this class is affected, but the file runs with failfast, so leaving it
# enabled also cuts off the class that sorts after it.
@unittest.skip(
    "flashinfer-ai/flashinfer#4486: on SM100/SM103 the TRTLLM_GEN tile-192 BMM "
    "path returns non-finite MoE output from FlashInfer 0.6.16.post4 on, so the "
    "first real prefill trips the sampler NaN assert and gsm8k scores 0.0. "
    "See #34629 for the package bisect."
)
class TestFlashinferTrtllmGenMoeBackendNvFp4Online(
    FlashinferNvFp4OnlineMoeBackendBase, CustomTestCase
):
    backend = "flashinfer_trtllm"
    model = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    eval_args = {"api": "completion", "max_tokens": 512}
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
    model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    extra_args = [
        "--reasoning-parser",
        "nemotron_3",
        "--tool-call-parser",
        "qwen3_coder",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]
    eval_args = {"max_tokens": 16000, "temperature": 1.0, "top_p": 0.95}
    spec_accept_length_threshold = 2.5
    extra_env = {
        "FLASHINFER_NVFP4_4OVER6": "1",
        "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
        "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
    }


if __name__ == "__main__":
    unittest.main()
