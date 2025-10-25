import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
    DEFAULT_MODEL_NAME_FOR_TEST_W8A8_WITH_MOE,
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestMoERunnerTriton(CustomTestCase):
    BASE_URL = DEFAULT_URL_FOR_TEST
    TIMEOUT = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    DEFAULT_MODEL = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT
    DEFAULT_EVAL_KWARGS = {
        "eval_name": "mmlu",
        "num_examples": 5,
        "num_threads": 1,
    }

    CONFIGS = {
        "moe_runner_triton_unquant_standard": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "triton",
                "--tp",
                "2",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.95",
            ],
        },
        "moe_runner_triton_kernel_unquant_standard": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "triton_kernel",
                "--tp",
                "2",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.95",
            ],
        },
        "moe_runner_deep_gemm_awq_quantization": {
            "model": DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "deep_gemm",
                "--quantization",
                "awq",
                "--tp",
                "2",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.95",
            ],
            "eval_kwargs": {"num_examples": 3},
        },
        "moe_runner_flashinfer_trtllm_mxfp4_quantization": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_trtllm",
                "--quantization",
                "mxfp4",
                "--tp",
                "2",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.95",
            ],
            "eval_kwargs": {"num_examples": 2},
        },
        "moe_runner_flashinfer_mxfp4_fp8_quantization": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_mxfp4",
                "--quantization",
                "mxfp4",
                "--tp",
                "2",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.95",
            ],
            "eval_kwargs": {"num_examples": 2},
        },
        "moe_runner_flashinfer_cutedsl_fp8_quantization": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_cutedsl",
                "--quantization",
                "fp8",
                "--tp",
                "2",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.95",
            ],
            "eval_kwargs": {"num_examples": 2},
        },
        "moe_runner_cutlass_w8a8_quantization": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_W8A8_WITH_MOE,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "cutlass",
                "--quantization",
                "w8a8",
                "--tp",
                "2",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.95",
            ],
            "eval_kwargs": {"num_examples": 2},
        },
    }

    def _run_config(self, config: dict) -> None:
        model = config.get("model", self.DEFAULT_MODEL)
        other_args = config.get("other_args", [])
        eval_kwargs = self.DEFAULT_EVAL_KWARGS | config.get("eval_kwargs", {})

        env_overrides = config.get("env", {})
        saved_env = {k: os.environ.get(k) for k in env_overrides}
        os.environ.update(env_overrides)

        process = popen_launch_server(
            model,
            self.BASE_URL,
            timeout=self.TIMEOUT,
            other_args=other_args,
        )
        try:
            args = SimpleNamespace(
                base_url=self.BASE_URL,
                model=model,
                **eval_kwargs,
            )
            metrics = run_eval(args)
            print(f"{metrics=}")
            self.assertGreaterEqual(metrics["score"], 0.0)
        finally:
            kill_process_tree(process.pid)
            for key, previous in saved_env.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous


def _make_test(config_name: str, config: dict):
    def test(self):
        self._run_config(config)

    test.__name__ = f"test_{config_name}"
    return test


for _name, _config in TestMoERunnerTriton.CONFIGS.items():
    setattr(TestMoERunnerTriton, f"test_{_name}", _make_test(_name, _config))


if __name__ == "__main__":
    unittest.main()
