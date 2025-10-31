import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
    DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
    DEFAULT_MODEL_NAME_FOR_TEST_W8A8_WITH_MOE,
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestMoERunner(CustomTestCase):
    BASE_URL = DEFAULT_URL_FOR_TEST
    TIMEOUT = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    DEFAULT_EVAL_KWARGS = {
        "eval_name": "mmlu",
        "num_examples": 5,
        "num_threads": 1,
    }

    CONFIGS = {
        "moe_runner_auto": {
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
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_triton": {
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
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_triton_kernel": {
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
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_flashinfer_cutlass": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,  # requires model with modelopt_fp4 quantization
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_cutlass",
                "--tp",
                "2",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.95",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_deep_gemm": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "deep_gemm",
                "--tp",
                "2",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.95",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_flashinfer_trtllm": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_trtllm",
                "--tp",
                "2",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.95",
            ],
        },
        "moe_runner_flashinfer_mxfp4": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
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
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_flashinfer_cutedsl": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_cutedsl",
                "--tp",
                "2",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.95",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_cutlass": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "cutlass",
                "--tp",
                "2",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.95",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
    }

    def _run_config(self, config: dict) -> None:
        model = config["model"]
        other_args = config.get("other_args", [])
        eval_kwargs = self.DEFAULT_EVAL_KWARGS

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
            self.assertGreaterEqual(metrics["score"], 0.48)
        finally:
            kill_process_tree(process.pid)


for _name, _cfg in TestMoERunner.CONFIGS.items():
    setattr(
        TestMoERunner,
        f"test_{_name}",
        (lambda self, cfg=_cfg: self._run_config(cfg)),
    )


if __name__ == "__main__":
    unittest.main()
