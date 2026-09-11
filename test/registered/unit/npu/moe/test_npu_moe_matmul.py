"""
Unit tests for sglang.srt.hardware_backend.npu.moe.matmul.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Mock optional third-party deps pulled in by sglang/__init__ when they are
# not installed, so these tests also run on a CPU-only box (CI installs them).
for _mod in ("triton", "IPython", "IPython.display", "aiohttp"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules.setdefault(_mod, MagicMock())

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")

# Mock NPU-only modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
):
    sys.modules.setdefault(_, MagicMock())

from sglang.srt.hardware_backend.npu.moe.matmul import (
    BaseMatmul,
    GroupedMatmul,
    GroupedMatmulSwigluQuant,
)

_PREFIX = "w13"
_MODULE = "sglang.srt.hardware_backend.npu.moe.matmul"


def _make_layer(weight_shape=(4, 8, 16)):
    """Layer with a real weight tensor."""
    return SimpleNamespace(**{f"{_PREFIX}_weight": torch.randn(*weight_shape)})


# =============================================================================
# BaseMatmul
# =============================================================================
class TestBaseMatmul(unittest.TestCase):
    def test_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseMatmul()

    def test_abstract_method_exists(self):
        self.assertTrue(hasattr(BaseMatmul, "forward"))


# =============================================================================
# GroupedMatmul
# =============================================================================
class TestGroupedMatmul(unittest.TestCase):
    def _common_args(self):
        return dict(
            layer=_make_layer(),
            weight_prefix=_PREFIX,
            hidden_states=torch.randn(8, 16),
            expert_tokens=torch.tensor([2, 2, 2, 2]),
            output_dtype=torch.float16,
            group_list_type=1,
            transposed=True,
        )

    @patch("torch.ops")
    def test_op_called(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul.return_value = [torch.randn(8, 16)]
        GroupedMatmul().forward(**self._common_args())
        mock_ops.npu.npu_grouped_matmul.assert_called_once()

    @patch("torch.ops")
    def test_weight_not_found_raises(self, mock_ops):
        layer = SimpleNamespace()  # no weight attr
        with self.assertRaises(AttributeError):
            GroupedMatmul().forward(**{**self._common_args(), "layer": layer})

    @patch("torch.ops")
    def test_x_passed_as_list(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul.return_value = [torch.randn(8, 16)]
        args = self._common_args()
        GroupedMatmul().forward(**args)
        kwargs = mock_ops.npu.npu_grouped_matmul.call_args.kwargs
        self.assertIsInstance(kwargs["x"], list)
        self.assertIs(kwargs["x"][0], args["hidden_states"])

    @patch("torch.ops")
    def test_weight_transposed_true(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul.return_value = [torch.randn(8, 16)]
        args = self._common_args()
        GroupedMatmul().forward(**args)
        kwargs = mock_ops.npu.npu_grouped_matmul.call_args.kwargs
        weight = getattr(args["layer"], f"{_PREFIX}_weight")
        self.assertIs(kwargs["weight"][0], weight)

    @patch("torch.ops")
    def test_weight_transposed_false(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul.return_value = [torch.randn(8, 16)]
        args = self._common_args()
        args["transposed"] = False
        GroupedMatmul().forward(**args)
        kwargs = mock_ops.npu.npu_grouped_matmul.call_args.kwargs
        weight = getattr(args["layer"], f"{_PREFIX}_weight")
        self.assertTrue(torch.equal(kwargs["weight"][0], weight.transpose(1, 2)))

    @patch("torch.ops")
    def test_fixed_kwargs(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul.return_value = [torch.randn(8, 16)]
        GroupedMatmul().forward(**self._common_args())
        kwargs = mock_ops.npu.npu_grouped_matmul.call_args.kwargs
        self.assertEqual(kwargs["split_item"], 2)
        self.assertEqual(kwargs["group_type"], 0)

    @patch("torch.ops")
    def test_group_list_and_output_dtype(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul.return_value = [torch.randn(8, 16)]
        args = self._common_args()
        GroupedMatmul().forward(**args)
        kwargs = mock_ops.npu.npu_grouped_matmul.call_args.kwargs
        self.assertIs(kwargs["group_list"], args["expert_tokens"])
        self.assertEqual(kwargs["output_dtype"], torch.float16)
        self.assertEqual(kwargs["group_list_type"], 1)

    @patch("torch.ops")
    def test_returns_first_element(self, mock_ops):
        expected = torch.randn(8, 16)
        mock_ops.npu.npu_grouped_matmul.return_value = [expected, "extra"]
        out = GroupedMatmul().forward(**self._common_args())
        self.assertIs(out, expected)

    @patch("torch.ops")
    def test_scale_args_passthrough(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul.return_value = [torch.randn(8, 16)]
        args = self._common_args()
        GroupedMatmul().forward(**args, weight_scale=torch.tensor(0.1))
        kwargs = mock_ops.npu.npu_grouped_matmul.call_args.kwargs
        self.assertIn("weight_scale", kwargs)


# =============================================================================
# GroupedMatmulSwigluQuant
# =============================================================================
class TestGroupedMatmulSwigluQuant(unittest.TestCase):
    def _common_args(self):
        return dict(
            layer=_make_layer(),
            weight_prefix=_PREFIX,
            hidden_states=torch.randn(8, 16),
            expert_tokens=torch.tensor([2, 2, 2, 2]),
        )

    @patch("torch.ops")
    def test_op_called(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.return_value = (
            torch.randn(8, 8),
            torch.tensor(0.5),
        )
        GroupedMatmulSwigluQuant().forward(**self._common_args())
        mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.assert_called_once()

    @patch("torch.ops")
    def test_weight_not_found_raises(self, mock_ops):
        layer = SimpleNamespace()
        with self.assertRaises(AttributeError):
            GroupedMatmulSwigluQuant().forward(
                **{**self._common_args(), "layer": layer}
            )

    @patch("torch.ops")
    def test_x_passed_directly(self, mock_ops):
        """x is a tensor, not a list (unlike GroupedMatmul)."""
        mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.return_value = (
            torch.randn(8, 8),
            torch.tensor(0.5),
        )
        args = self._common_args()
        GroupedMatmulSwigluQuant().forward(**args)
        kwargs = mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.call_args.kwargs
        self.assertIs(kwargs["x"], args["hidden_states"])
        self.assertNotIsInstance(kwargs["x"], list)

    @patch("torch.ops")
    def test_weight_transposed_true(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.return_value = (
            torch.randn(8, 8),
            torch.tensor(0.5),
        )
        args = self._common_args()
        GroupedMatmulSwigluQuant().forward(**args)
        kwargs = mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.call_args.kwargs
        weight = getattr(args["layer"], f"{_PREFIX}_weight")
        self.assertIs(kwargs["weight"][0], weight)

    @patch("torch.ops")
    def test_weight_transposed_false(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.return_value = (
            torch.randn(8, 8),
            torch.tensor(0.5),
        )
        args = self._common_args()
        GroupedMatmulSwigluQuant().forward(**args, transposed=False)
        kwargs = mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.call_args.kwargs
        weight = getattr(args["layer"], f"{_PREFIX}_weight")
        self.assertTrue(torch.equal(kwargs["weight"][0], weight.transpose(1, 2)))

    @patch("torch.ops")
    def test_cumsum_when_group_list_type_1(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.return_value = (
            torch.randn(8, 8),
            torch.tensor(0.5),
        )
        expert_tokens = torch.tensor([2, 2, 2, 2])
        args = {**self._common_args(), "expert_tokens": expert_tokens}
        GroupedMatmulSwigluQuant().forward(**args, group_list_type=1)
        kwargs = mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.call_args.kwargs
        self.assertTrue(torch.equal(kwargs["group_list"], expert_tokens.cumsum(0)))

    @patch("torch.ops")
    def test_no_cumsum_when_group_list_type_not_1(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.return_value = (
            torch.randn(8, 8),
            torch.tensor(0.5),
        )
        expert_tokens = torch.tensor([2, 2, 2, 2])
        args = {**self._common_args(), "expert_tokens": expert_tokens}
        GroupedMatmulSwigluQuant().forward(**args, group_list_type=0)
        kwargs = mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.call_args.kwargs
        self.assertIs(kwargs["group_list"], expert_tokens)

    @patch("torch.ops")
    def test_returns_tuple_directly(self, mock_ops):
        expected = (torch.randn(8, 8), torch.tensor(0.5))
        mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.return_value = expected
        out = GroupedMatmulSwigluQuant().forward(**self._common_args())
        self.assertIs(out, expected)

    @patch("torch.ops")
    def test_scale_args_passthrough(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.return_value = (
            torch.randn(8, 8),
            torch.tensor(0.5),
        )
        GroupedMatmulSwigluQuant().forward(
            **self._common_args(), quant_dtype=torch.int8
        )
        kwargs = mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.call_args.kwargs
        self.assertIn("quant_dtype", kwargs)

    @patch("torch.ops")
    def test_defaults(self, mock_ops):
        mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.return_value = (
            torch.randn(8, 8),
            torch.tensor(0.5),
        )
        GroupedMatmulSwigluQuant().forward(**self._common_args())
        kwargs = mock_ops.npu.npu_grouped_matmul_swiglu_quant_v2.call_args.kwargs
        # default group_list_type=1 → cumsum applied
        expert_tokens = self._common_args()["expert_tokens"]
        self.assertTrue(torch.equal(kwargs["group_list"], expert_tokens.cumsum(0)))


if __name__ == "__main__":
    unittest.main()
