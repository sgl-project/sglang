import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=500, suite="nightly-4-gpu-b200", nightly=True)


class _FlashinferTrtllmGenMoeBackendMixin:
    MODEL_PATH = ""
    BACKENDS = ()
    EXTRA_OTHER_ARGS = ()
    ENV_OVERRIDES = None
    MIN_GSM8K_ACCURACY = 0.93

    def _run_gsm8k_for_backend(self, moe_runner_backend: str):
        launch_kwargs = {}
        if self.ENV_OVERRIDES is not None:
            launch_kwargs["env"] = {**os.environ, **self.ENV_OVERRIDES}

        process = popen_launch_server(
            self.MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--moe-runner-backend",
                moe_runner_backend,
                *self.EXTRA_OTHER_ARGS,
            ],
            **launch_kwargs,
        )
        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(DEFAULT_URL_FOR_TEST.split(":")[-1]),
            )
            metrics = run_eval(args)
            print(f"{moe_runner_backend=} {metrics=}")
            self.assertGreater(metrics["accuracy"], self.MIN_GSM8K_ACCURACY)
        finally:
            kill_process_tree(process.pid)

    def test_gsm8k(self):
        for moe_runner_backend in self.BACKENDS:
            with self.subTest(moe_runner_backend=moe_runner_backend):
                self._run_gsm8k_for_backend(moe_runner_backend)


class TestFlashinferTrtllmGenMoeBackendFP8(
    _FlashinferTrtllmGenMoeBackendMixin, CustomTestCase
):
    MODEL_PATH = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    BACKENDS = ("flashinfer_trtllm", "flashinfer_trtllm_routed")
    EXTRA_OTHER_ARGS = (
        "--attention-backend",
        "triton",
        "--tp-size",
        "4",
        "--ep-size",
        "4",
        "--mem-fraction-static",
        "0.7",
        "--mamba-ssm-dtype",
        "bfloat16",
    )
    ENV_OVERRIDES = {"SGLANG_ENABLE_JIT_DEEPGEMM": "False"}


class TestFlashinferTrtllmGenMoeBackendBF16(
    _FlashinferTrtllmGenMoeBackendMixin, CustomTestCase
):
    MODEL_PATH = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    BACKENDS = ("flashinfer_trtllm",)
    EXTRA_OTHER_ARGS = (
        "--attention-backend",
        "triton",
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
    )


class TestFlashinferTrtllmGenMoeBackendMXFP8(
    _FlashinferTrtllmGenMoeBackendMixin, CustomTestCase
):
    MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    BACKENDS = ("flashinfer_trtllm", "flashinfer_trtllm_routed")
    EXTRA_OTHER_ARGS = (
        "--quantization",
        "mxfp8",
        "--tp-size",
        "4",
        "--ep-size",
        "4",
        "--mem-fraction-static",
        "0.7",
    )
    ENV_OVERRIDES = {"SGLANG_ENABLE_JIT_DEEPGEMM": "False"}


if __name__ == "__main__":
    unittest.main()
