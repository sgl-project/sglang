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
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
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
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
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
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_flashinfer_trtllm_mxfp4_quantization": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
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
        },
        # Speculative decoding (NGRAM) with FlashInfer MXFP4 backend (differs from main path)
        "moe_runner_flashinfer_mxfp4_quantization_spec_ngram": {
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
                "--speculative-algorithm",
                "NGRAM",
                "--speculative-num-draft-tokens",
                "8",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_flashinfer_mxfp4_quantization": {
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
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_cutlass_w8a8_quantization": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_W8A8_WITH_MOE,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "cutlass",
                "--quantization",
                "w8a8_int8",
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
        "moe_runner_auto_refactored": {  # 'auto' where the potential backend has been refactored
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
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
        "moe_runner_auto_not_refactored": {  # 'auto' where the potential backend has not been refactored
            "model": DEFAULT_MODEL_NAME_FOR_TEST_W8A8_WITH_MOE,
            "other_args": [
                "--trust-remote-code",
                "--quantization",
                "w8a8_int8",
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
