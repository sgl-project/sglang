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


class FlashinferTrtllmGenMoeBackendFP8Base:
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, "SGLANG_ENABLE_JIT_DEEPGEMM": "False"},
            other_args=[
                "--attention-backend",
                "triton",
                "--moe-runner-backend",
                cls.backend,
                "--tp-size",
                "4",
                "--ep-size",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--mamba-ssm-dtype",
                "bfloat16",
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
        self.assertGreater(metrics["score"], 0.89)


class FlashinferTrtllmGenMoeBackendBF16Base:
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-Next-80B-A3B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "triton",
                "--moe-runner-backend",
                cls.backend,
                "--cuda-graph-max-bs-decode",
                "512",
                "--tp-size",
                "4",
                "--ep-size",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--mamba-ssm-dtype",
                "bfloat16",
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
        self.assertGreater(metrics["score"], 0.93)


class FlashinferTrtllmGenMoeBackendMXFP8Base:
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.model = "zianglih/Qwen3-30B-A3B-Instruct-2507-MXFP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, "SGLANG_ENABLE_JIT_DEEPGEMM": "False"},
            other_args=[
                "--fp8-gemm-backend",
                "flashinfer_cutlass",
                "--moe-runner-backend",
                cls.backend,
                "--tp-size",
                "4",
                "--ep-size",
                "4",
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
        self.assertGreater(metrics["score"], 0.93)


class FlashinferTrtllmGenMoeBackendMXFP8MixedBF16Base:
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.model = "zianglih/JoyAI-LLM-Flash-MXFP8-last-6-BF16"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, "SGLANG_ENABLE_JIT_DEEPGEMM": "False"},
            other_args=[
                "--kv-cache-dtype",
                "bf16",
                "--fp8-gemm-backend",
                "flashinfer_trtllm",
                "--moe-runner-backend",
                cls.backend,
                "--tp-size",
                "4",
                "--trust-remote-code",
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
        self.assertGreater(metrics["score"], 0.92)


class FlashinferTrtllmGenMoeBackendNVFP4Base:
    backend = None
    extra_env = {}

    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/Qwen3-30B-A3B-NVFP4"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, **cls.extra_env, "SGLANG_ENABLE_JIT_DEEPGEMM": "False"},
            other_args=[
                "--moe-runner-backend",
                cls.backend,
                "--tp-size",
                "4",
                "--ep-size",
                "4",
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
        self.assertGreater(metrics["score"], 0.89)


class FlashinferTrtllmGenMoeBackendNvFp4OnlineBase:
    backend = None
    extra_env = {}

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
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
                "2",
                "--quantization",
                "nvfp4_online",
                "--mem-fraction-static",
                "0.7",
                "--mamba-ssm-dtype",
                "bfloat16",
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


class TestFlashinferTrtllmGenMoeBackendFP8(
    FlashinferTrtllmGenMoeBackendFP8Base, CustomTestCase
):
    backend = "flashinfer_trtllm"


class TestFlashinferTrtllmGenMoeBackendNVFP4(
    FlashinferTrtllmGenMoeBackendNVFP4Base, CustomTestCase
):
    backend = "flashinfer_trtllm"


class TestFlashinferTrtllmGenMoeBackendMXFP8Routed(
    FlashinferTrtllmGenMoeBackendMXFP8Base, CustomTestCase
):
    backend = "flashinfer_trtllm_routed"


class TestFlashinferTrtllmRoutedMxfp8MixedBF16(
    FlashinferTrtllmGenMoeBackendMXFP8MixedBF16Base, CustomTestCase
):
    backend = "flashinfer_trtllm_routed"


class TestFlashinferTrtllmGenMoeBackendBF16Routed(
    FlashinferTrtllmGenMoeBackendBF16Base, CustomTestCase
):
    backend = "flashinfer_trtllm_routed"


class TestFlashinferTrtllmGenMoeBackendNvFp4PerTokenActivationRouted(
    FlashinferTrtllmGenMoeBackendNVFP4Base, CustomTestCase
):
    extra_env = {"SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1"}
    backend = "flashinfer_trtllm_routed"


class TestFlashinferTrtllmGenMoeBackendNvFp4Online(
    FlashinferTrtllmGenMoeBackendNvFp4OnlineBase, CustomTestCase
):
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
    backend = "flashinfer_trtllm"


if __name__ == "__main__":
    unittest.main()
