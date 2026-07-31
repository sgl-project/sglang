"""Shared fixture for FlashInfer online-NVFP4 MoE backend tests.

Concrete tests combine these pure mixins with ``CustomTestCase`` and select
the target/draft runner and A2A backends explicitly. The server-info checks
prove that the requested target and speculative configurations survived
argument resolution; GSM8K and speculative acceptance then exercise both
models end to end.
"""

import os
import statistics
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class FlashinferNvFp4OnlineMoeBackendBase:
    backend: str | None = None
    model: str | None = None
    quantization = "nvfp4_online"
    extra_args: list[str] = []
    extra_env: dict[str, str] = {}
    eval_args: dict[str, object] = {}
    expected_server_args: dict[str, object] = {}
    spec_accept_length_threshold: float | None = None
    enable_jit_deepgemm = False

    @classmethod
    def setUpClass(cls):
        assert cls.backend is not None, f"{cls.__name__} must set `backend`"
        assert cls.model is not None, f"{cls.__name__} must set `model`"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={
                **os.environ,
                **cls.extra_env,
                "SGLANG_ENABLE_JIT_DEEPGEMM": str(cls.enable_jit_deepgemm),
            },
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
                cls.quantization,
                "--mem-fraction-static",
                "0.7",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        server_info_response = requests.get(self.base_url + "/server_info", timeout=30)
        server_info_response.raise_for_status()
        server_info = server_info_response.json()
        for name, expected in self.expected_server_args.items():
            self.assertEqual(
                server_info.get(name),
                expected,
                f"resolved server argument mismatch for {name}",
            )

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
            server_info_response = requests.get(
                self.base_url + "/server_info", timeout=30
            )
            server_info_response.raise_for_status()
            server_info = server_info_response.json()
            spec_accept_lengths = [
                state["avg_spec_accept_length"]
                for state in server_info["internal_states"]
                if "avg_spec_accept_length" in state
            ]
            self.assertEqual(len(spec_accept_lengths), server_info["dp_size"])
            avg_spec_accept_length = statistics.fmean(spec_accept_lengths)
            print(f"{spec_accept_lengths=}, {avg_spec_accept_length=}")
            self.assertGreater(
                avg_spec_accept_length, self.spec_accept_length_threshold
            )


class NemotronNvFp4OnlineMoeBackendBase(FlashinferNvFp4OnlineMoeBackendBase):
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
        "--speculative-moe-runner-backend",
        "flashinfer_cutedsl",
        "--speculative-draft-model-quantization",
        "nvfp4_online",
    ]
    eval_args = {"max_tokens": 16000, "temperature": 1.0, "top_p": 0.95}
    spec_accept_length_threshold = 2.5
    extra_env = {
        "FLASHINFER_NVFP4_4OVER6": "1",
        "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
        "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
    }
