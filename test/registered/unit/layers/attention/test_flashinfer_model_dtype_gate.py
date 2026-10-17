import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    _validate_model_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashInferModelDtypeGate(CustomTestCase):
    """--dtype float32 is accepted by the argument parser, but FlashInfer has
    no fp32 attention kernel, so the engine used to die with a bare KeyError
    from the JIT dispatch table during CUDA-graph capture. The backend now
    rejects the dtype at construction with an actionable message."""

    def test_supported_dtypes_pass(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                _validate_model_dtype(dtype)

    def test_float32_rejected_with_guidance(self):
        for dtype in (torch.float32, torch.float):
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError) as ctx:
                    _validate_model_dtype(dtype)
                message = str(ctx.exception)
                self.assertIn("float16", message)
                self.assertIn("bfloat16", message)
                self.assertIn("torch_native", message)

    def test_backend_init_gates_before_any_state(self):
        # A bare runner mock carries no pools/config; reaching anything past
        # the dtype gate would raise AttributeError instead of ValueError.
        runner = MagicMock(dtype=torch.float32)
        with self.assertRaises(ValueError) as ctx:
            FlashInferAttnBackend(runner)
        self.assertIn("float32", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
