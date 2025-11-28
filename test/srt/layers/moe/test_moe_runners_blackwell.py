import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,
    DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestMoERunnerBlackwell(CustomTestCase):
    """Tests for MoE runner backends that require Blackwell (SM100+) GPUs.

    These tests use NVFP4 or MXFP4 quantized models which require native
    FP4 tensor core support only available on Blackwell architecture.
    """

    BASE_URL = DEFAULT_URL_FOR_TEST
    TIMEOUT = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    DEFAULT_EVAL_KWARGS = {
        "eval_name": "mmlu",
        "num_examples": 5,
        "num_threads": 1,
    }

    CONFIGS = {
        # NVFP4 models require Blackwell (is_blackwell_supported() check in modelopt_quant.py)
        "moe_runner_flashinfer_cutlass": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_cutlass",
            ],
        },
        "moe_runner_flashinfer_cutedsl": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_cutedsl",
            ],
        },
        "moe_runner_cutlass": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "cutlass",
            ],
        },
        # MXFP4 models require Blackwell for native FP4 support
        "moe_runner_flashinfer_mxfp4": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_mxfp4",
                "--quantization",
                "mxfp4",
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


for _name, _cfg in TestMoERunnerBlackwell.CONFIGS.items():
    setattr(
        TestMoERunnerBlackwell,
        f"test_{_name}",
        (lambda self, cfg=_cfg: self._run_config(cfg)),
    )


if __name__ == "__main__":
    unittest.main()
