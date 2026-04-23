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

register_cuda_ci(est_time=600, suite="nightly-4-gpu-b200", nightly=True)


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
                "--cuda-graph-max-bs",
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
        cls.model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, "SGLANG_ENABLE_JIT_DEEPGEMM": "False"},
            other_args=[
                "--quantization",
                "mxfp8",
                "--fp8-gemm-backend",
                "flashinfer_trtllm",
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


class FlashinferTrtllmGenMoeBackendNVFP4Base:
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/Qwen3-30B-A3B-NVFP4"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, "SGLANG_ENABLE_JIT_DEEPGEMM": "False"},
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


class TestFlashinferTrtllmGenMoeBackendFP8(
    FlashinferTrtllmGenMoeBackendFP8Base, CustomTestCase
):
    backend = "flashinfer_trtllm"


class TestFlashinferTrtllmGenMoeBackendMXFP8(
    FlashinferTrtllmGenMoeBackendMXFP8Base, CustomTestCase
):
    backend = "flashinfer_trtllm"


class TestFlashinferTrtllmGenMoeBackendBF16(
    FlashinferTrtllmGenMoeBackendBF16Base, CustomTestCase
):
    backend = "flashinfer_trtllm"


class TestFlashinferTrtllmGenMoeBackendNVFP4(
    FlashinferTrtllmGenMoeBackendNVFP4Base, CustomTestCase
):
    backend = "flashinfer_trtllm"


class TestFlashinferTrtllmGenMoeBackendFP8Routed(
    FlashinferTrtllmGenMoeBackendFP8Base, CustomTestCase
):
    backend = "flashinfer_trtllm_routed"


class TestFlashinferTrtllmGenMoeBackendMXFP8Routed(
    FlashinferTrtllmGenMoeBackendMXFP8Base, CustomTestCase
):
    backend = "flashinfer_trtllm_routed"


class TestFlashinferTrtllmGenMoeBackendBF16Routed(
    FlashinferTrtllmGenMoeBackendBF16Base, CustomTestCase
):
    backend = "flashinfer_trtllm_routed"


class TestFlashinferTrtllmGenMoeBackendNVFP4Routed(
    FlashinferTrtllmGenMoeBackendNVFP4Base, CustomTestCase
):
    backend = "flashinfer_trtllm_routed"


if __name__ == "__main__":
    unittest.main()
