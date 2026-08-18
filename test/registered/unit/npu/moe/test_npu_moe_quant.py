"""
Unit tests for sglang.srt.hardware_backend.npu.moe.quant.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")

# Mock NPU-only modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
):
    sys.modules.setdefault(_, MagicMock())

from sglang.srt.hardware_backend.npu.moe.quant import (
    BaseHiddenStatesQuant,
    HiddenStatesDynamicQuant,
    HiddenStatesStaticQuant,
)


# =============================================================================
# BaseHiddenStatesQuant
# =============================================================================
class TestBaseHiddenStatesQuant(unittest.TestCase):
    def test_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseHiddenStatesQuant(torch.int8)

    def test_abstract_method_exists(self):
        self.assertTrue(hasattr(BaseHiddenStatesQuant, "__call__"))


# =============================================================================
# HiddenStatesDynamicQuant — init
# =============================================================================
class TestHiddenStatesDynamicQuantInit(unittest.TestCase):
    @patch("torch.ops")
    def test_float8_uses_mx_quant(self, mock_ops):
        quant = HiddenStatesDynamicQuant(torch.float8_e4m3fn)
        self.assertIs(quant._op, mock_ops.npu.npu_dynamic_mx_quant)

    @patch("torch.ops")
    def test_int8_uses_dynamic_quant(self, mock_ops):
        quant = HiddenStatesDynamicQuant(torch.int8)
        self.assertIs(quant._op, mock_ops.npu.npu_dynamic_quant)

    @patch("torch.ops")
    def test_quint4x2_uses_dynamic_quant(self, mock_ops):
        quant = HiddenStatesDynamicQuant(torch.quint4x2)
        self.assertIs(quant._op, mock_ops.npu.npu_dynamic_quant)

    @patch("torch.ops")
    def test_unsupported_dtype_raises(self, mock_ops):
        with self.assertRaises(ValueError):
            HiddenStatesDynamicQuant(torch.float32)

    @patch("torch.ops")
    def test_quant_dtype_stored(self, mock_ops):
        quant = HiddenStatesDynamicQuant(torch.int8)
        self.assertEqual(quant.quant_dtype, torch.int8)


# =============================================================================
# HiddenStatesDynamicQuant — call
# =============================================================================
class TestHiddenStatesDynamicQuantCall(unittest.TestCase):
    def _make_quant(self, dtype=torch.int8):
        quant = HiddenStatesDynamicQuant(dtype)
        quant._op = MagicMock(
            return_value=(torch.randn(4, 8), torch.tensor(0.5))
        )
        return quant

    @patch("torch.ops")
    def test_calls_op(self, mock_ops):
        quant = self._make_quant()
        hidden = torch.randn(4, 8)
        quant(hidden)
        quant._op.assert_called_once_with(hidden, dst_type=torch.int8)

    @patch("torch.ops")
    def test_returns_tuple(self, mock_ops):
        quant = self._make_quant()
        out, scale = quant(torch.randn(4, 8))
        self.assertIsNotNone(out)
        self.assertIsNotNone(scale)

    @patch("torch.ops")
    def test_returns_op_output(self, mock_ops):
        expected_q = torch.randn(4, 8)
        expected_s = torch.tensor(0.5)
        quant = self._make_quant()
        quant._op = MagicMock(
            return_value=(expected_q, expected_s)
        )
        out, scale = quant(torch.randn(4, 8))
        self.assertIs(out, expected_q)
        self.assertIs(scale, expected_s)


# =============================================================================
# HiddenStatesStaticQuant
# =============================================================================
class TestHiddenStatesStaticQuant(unittest.TestCase):
    def _make_layer(self, with_attrs=True):
        layer = SimpleNamespace()
        if with_attrs:
            layer.aclnn_input_scale_reciprocal = torch.tensor(0.1)
            layer.aclnn_input_offset = torch.tensor(1.0)
        return layer

    @patch("torch.ops")
    def test_calls_npu_quantize(self, mock_ops):
        mock_ops.npu.npu_quantize.return_value = torch.randn(4, 8)
        quant = HiddenStatesStaticQuant(torch.int8)
        layer = self._make_layer()
        quant(torch.randn(4, 8), layer)
        mock_ops.npu.npu_quantize.assert_called_once()

    @patch("torch.ops")
    def test_op_args(self, mock_ops):
        mock_ops.npu.npu_quantize.return_value = torch.randn(4, 8)
        hidden = torch.randn(4, 8)
        scale = torch.tensor(0.1)
        offset = torch.tensor(1.0)
        layer = SimpleNamespace(
            aclnn_input_scale_reciprocal=scale,
            aclnn_input_offset=offset,
        )
        quant = HiddenStatesStaticQuant(torch.int8)
        quant(hidden, layer)

        args = mock_ops.npu.npu_quantize.call_args.args
        self.assertIs(args[0], hidden)
        self.assertIs(args[1], scale)
        self.assertIs(args[2], offset)
        self.assertEqual(args[3], torch.int8)
        self.assertEqual(args[4], -1)
        self.assertFalse(args[5])

    @patch("torch.ops")
    def test_returns_quantized_and_none(self, mock_ops):
        expected = torch.randn(4, 8)
        mock_ops.npu.npu_quantize.return_value = expected
        quant = HiddenStatesStaticQuant(torch.int8)
        out, scale = quant(torch.randn(4, 8), self._make_layer())
        self.assertIs(out, expected)
        self.assertIsNone(scale)

    @patch("torch.ops")
    def test_missing_scale_raises(self, mock_ops):
        quant = HiddenStatesStaticQuant(torch.int8)
        layer = SimpleNamespace(aclnn_input_offset=torch.tensor(1.0))
        with self.assertRaises(AttributeError):
            quant(torch.randn(4, 8), layer)

    @patch("torch.ops")
    def test_missing_offset_raises(self, mock_ops):
        quant = HiddenStatesStaticQuant(torch.int8)
        layer = SimpleNamespace(
            aclnn_input_scale_reciprocal=torch.tensor(0.1)
        )
        with self.assertRaises(AttributeError):
            quant(torch.randn(4, 8), layer)

    @patch("torch.ops")
    def test_missing_both_raises(self, mock_ops):
        quant = HiddenStatesStaticQuant(torch.int8)
        layer = SimpleNamespace()
        with self.assertRaises(AttributeError):
            quant(torch.randn(4, 8), layer)


if __name__ == "__main__":
    unittest.main()
