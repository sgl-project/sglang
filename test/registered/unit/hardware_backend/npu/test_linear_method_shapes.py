"""Check W8A8 tensor layout handling; replace only the device operators."""

import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

# Initialize the quantization registry before importing its backend methods.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW8A8Int8DynamicLinearMethod,
    NPUW8A8Int8LinearMethod,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@contextmanager
def _mock_operator_modules(modules):
    # Restore only the operator keys; torch may lazily load modules in this scope.
    missing = object()
    original = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in original.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class TestNpuLinearShapes(CustomTestCase):
    def quantize(self, x, *args):
        self.assertEqual(x.ndim, 2)
        return x.to(torch.int8)

    def dynamic_quantize(self, x):
        return self.quantize(x), torch.ones(x.shape[0])

    def matmul(self, x, weight, scale, *, bias=None, output_dtype, **kwargs):
        self.assertEqual(x.ndim, 2)
        result = x.float() @ weight.float()
        if bias is not None:
            result += bias
        return result.to(output_dtype)

    def setUp(self):
        super().setUp()
        # Use width 4 -> 6 so restoring the input shape would fail.
        self.layer = SimpleNamespace(
            weight=(torch.arange(24).reshape(4, 6) % 3).to(torch.int8),
            weight_scale=torch.ones(6),
            deq_scale=torch.ones(6),
            aclnn_input_scale_reciprocal=torch.ones(4),
            aclnn_input_offset=torch.zeros(4),
            quant_bias=torch.ones(6),
        )
        self.enterContext(
            patch.object(
                torch.ops.npu, "npu_quantize", side_effect=self.quantize, create=True
            )
        )
        self.dynamic_quant = MagicMock(side_effect=self.dynamic_quantize)
        operators = SimpleNamespace(
            quant=SimpleNamespace(dynamic_quant=self.dynamic_quant),
            matmul=SimpleNamespace(quant_matmul=MagicMock(side_effect=self.matmul)),
        )
        self.enterContext(
            _mock_operator_modules({"sgl_kernel_npu.operators": operators})
        )

    def test_static_and_dynamic_restore_leading_dimensions(self):
        """Keep leading dimensions, e.g. (3, 2, 4) -> (3, 2, 6), and dtype."""
        for method in (NPUW8A8Int8LinearMethod(), NPUW8A8Int8DynamicLinearMethod()):
            for shape in ((3, 4), (2, 3, 4), (2, 1, 3, 4)):
                with self.subTest(method=type(method).__name__, shape=shape):
                    x = (
                        torch.arange(torch.tensor(shape).prod()).reshape(shape) % 3
                    ).to(torch.bfloat16)
                    if x.ndim > 2:
                        x = x.transpose(0, 1)  # Also exercise noncontiguous inputs.
                    result = method.apply(self.layer, x, bias=torch.ones(6))
                    expected = (x.float() @ self.layer.weight.float() + 1).to(x.dtype)
                    self.assertEqual(result.shape, (*x.shape[:-1], 6))
                    self.assertEqual(result.dtype, x.dtype)
                    torch.testing.assert_close(result, expected)

    def test_prequantized_inputs_keep_existing_paths(self):
        """INT8 or (INT8, scale) inputs must not be quantized again."""
        x = torch.ones(2, 3, 4, dtype=torch.int8)
        result = NPUW8A8Int8LinearMethod().apply(self.layer, x)
        torch.testing.assert_close(
            result, (x.float() @ self.layer.weight.float() + 1).to(torch.int8)
        )
        torch.ops.npu.npu_quantize.assert_not_called()

        x = x.reshape(-1, 4)
        result = NPUW8A8Int8DynamicLinearMethod().apply(
            self.layer, (x, torch.ones(2, 3))
        )
        torch.testing.assert_close(
            result, (x.float() @ self.layer.weight.float()).to(torch.bfloat16)
        )
        self.dynamic_quant.assert_not_called()


if __name__ == "__main__":
    unittest.main()
