"""
Unit tests for sglang.srt.hardware_backend.npu.moe.init_routing.
"""

import sys
import unittest
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

from sglang.srt.hardware_backend.npu.moe.init_routing import (
    MXFP8_QUANT_MODE,
    BaseInitRouting,
    NPUMoEInitRouting_Quant,
    NPUMoEInitRouting_v1,
    NPUMoEInitRouting_v2,
    _normalize_mxfp_scale,
)


# =============================================================================
# MXFP8_QUANT_MODE
# =============================================================================
class TestMXFP8QuantMode(unittest.TestCase):
    def test_value(self):
        self.assertEqual(MXFP8_QUANT_MODE, 3)


# =============================================================================
# _normalize_mxfp_scale  (pure torch)
# =============================================================================
class TestNormalizeMxfpScale(unittest.TestCase):
    def test_2d_reshaped(self):
        scale = torch.randn(4, 8)
        out = _normalize_mxfp_scale(scale)
        self.assertEqual(out.shape, (4, 4, 2))

    def test_none_passthrough(self):
        self.assertIsNone(_normalize_mxfp_scale(None))

    def test_3d_passthrough(self):
        scale = torch.randn(4, 4, 2)
        out = _normalize_mxfp_scale(scale)
        self.assertIs(out, scale)

    def test_1d_passthrough(self):
        scale = torch.randn(8)
        out = _normalize_mxfp_scale(scale)
        self.assertIs(out, scale)

    def test_values_preserved(self):
        scale = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        out = _normalize_mxfp_scale(scale)
        self.assertTrue(torch.equal(out.flatten(), scale.flatten()))


# =============================================================================
# BaseInitRouting
# =============================================================================
class TestBaseInitRouting(unittest.TestCase):
    def test_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseInitRouting()

    def test_abstract_method_exists(self):
        self.assertTrue(hasattr(BaseInitRouting, "_init_routing"))


# =============================================================================
# NPUMoEInitRouting_v1
# =============================================================================
class TestNPUMoEInitRoutingV1(unittest.TestCase):
    def _setup_mocks(self, mock_ops):
        mock_ops.npu.npu_moe_init_routing.return_value = (
            torch.randn(8, 16),
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]),
        )
        mock_ops.npu.npu_moe_compute_expert_tokens.return_value = torch.tensor(
            [4, 4, 0, 0]
        )

    @patch("torch.ops")
    def test_row_idx_construction(self, mock_ops):
        """row_idx = arange(N*K).view(K, -1).permute(1,0).contiguous()."""
        self._setup_mocks(mock_ops)
        hidden = torch.randn(4, 16)
        topk_ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])

        act = NPUMoEInitRouting_v1()
        act._init_routing(hidden, topk_ids, num_experts=4, top_k=2)

        _, kwargs = mock_ops.npu.npu_moe_init_routing.call_args
        row_idx = kwargs["row_idx"]
        expected = (
            torch.arange(0, 8, dtype=torch.int32)
            .view(2, -1)
            .permute(1, 0)
            .contiguous()
        )
        self.assertTrue(torch.equal(row_idx, expected))

    @patch("torch.ops")
    def test_row_idx_shape(self, mock_ops):
        self._setup_mocks(mock_ops)
        hidden = torch.randn(4, 16)
        topk_ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])

        act = NPUMoEInitRouting_v1()
        act._init_routing(hidden, topk_ids, num_experts=4, top_k=2)

        _, kwargs = mock_ops.npu.npu_moe_init_routing.call_args
        self.assertEqual(kwargs["row_idx"].shape, (4, 2))

    @patch("torch.ops")
    def test_init_routing_kwargs(self, mock_ops):
        self._setup_mocks(mock_ops)
        hidden = torch.randn(4, 16)
        topk_ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])

        act = NPUMoEInitRouting_v1()
        act._init_routing(hidden, topk_ids, num_experts=4, top_k=2)

        args, kwargs = mock_ops.npu.npu_moe_init_routing.call_args
        self.assertIs(args[0], hidden)
        self.assertIs(kwargs["expert_idx"], topk_ids)
        self.assertEqual(kwargs["active_num"], 4)

    @patch("torch.ops")
    def test_compute_expert_tokens_called(self, mock_ops):
        self._setup_mocks(mock_ops)
        expanded_expert_idx = mock_ops.npu.npu_moe_init_routing.return_value[2]

        act = NPUMoEInitRouting_v1()
        act._init_routing(
            torch.randn(4, 16),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
            num_experts=4,
            top_k=2,
        )

        mock_ops.npu.npu_moe_compute_expert_tokens.assert_called_once_with(
            expanded_expert_idx, 4
        )

    @patch("torch.ops")
    def test_returns_none_scale(self, mock_ops):
        self._setup_mocks(mock_ops)
        act = NPUMoEInitRouting_v1()
        _, _, _, scale = act._init_routing(
            torch.randn(4, 16),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
            num_experts=4,
            top_k=2,
        )
        self.assertIsNone(scale)

    @patch("torch.ops")
    def test_expert_tokens_int64(self, mock_ops):
        self._setup_mocks(mock_ops)
        act = NPUMoEInitRouting_v1()
        _, _, expert_tokens, _ = act._init_routing(
            torch.randn(4, 16),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
            num_experts=4,
            top_k=2,
        )
        self.assertEqual(expert_tokens.dtype, torch.int64)


# =============================================================================
# NPUMoEInitRouting_v2 — init
# =============================================================================
class TestNPUMoEInitRoutingV2Init(unittest.TestCase):
    def test_default_quant_mode(self):
        act = NPUMoEInitRouting_v2()
        self.assertEqual(act.quant_mode, -1)

    def test_custom_quant_mode(self):
        act = NPUMoEInitRouting_v2(quant_mode=3)
        self.assertEqual(act.quant_mode, 3)


# =============================================================================
# NPUMoEInitRouting_v2 — apply
# =============================================================================
class TestNPUMoEInitRoutingV2Apply(unittest.TestCase):
    def _setup_mock(self, mock_ops, scale=None):
        mock_ops.npu.npu_moe_init_routing_v2.return_value = (
            torch.randn(8, 16),
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            torch.tensor([4, 4, 0, 0]),
            scale,
        )

    @patch("torch.ops")
    def test_op_kwargs(self, mock_ops):
        self._setup_mock(mock_ops)
        hidden = torch.randn(4, 16)
        topk_ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])

        act = NPUMoEInitRouting_v2()
        act._init_routing(hidden, topk_ids, num_experts=4, top_k=2)

        args, kwargs = mock_ops.npu.npu_moe_init_routing_v2.call_args
        self.assertIs(args[0], hidden)
        self.assertIs(args[1], topk_ids)
        self.assertEqual(kwargs["active_num"], 8)
        self.assertEqual(kwargs["expert_num"], 4)
        self.assertTrue(kwargs["expert_tokens_num_flag"])
        self.assertEqual(kwargs["active_expert_range"], [0, 4])
        self.assertEqual(kwargs["quant_mode"], -1)

    @patch("torch.ops")
    def test_quant_mode_neg1_returns_none_scale(self, mock_ops):
        self._setup_mock(mock_ops, scale=torch.randn(8, 8))
        act = NPUMoEInitRouting_v2(quant_mode=-1)
        _, _, _, scale = act._init_routing(
            torch.randn(4, 16),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
            num_experts=4,
            top_k=2,
        )
        self.assertIsNone(scale)

    @patch("torch.ops")
    def test_quant_mode_mxfp8_normalizes_scale(self, mock_ops):
        raw_scale = torch.randn(4, 8)
        self._setup_mock(mock_ops, scale=raw_scale)
        act = NPUMoEInitRouting_v2(quant_mode=MXFP8_QUANT_MODE)
        _, _, _, scale = act._init_routing(
            torch.randn(4, 16),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
            num_experts=4,
            top_k=2,
        )
        self.assertEqual(scale.shape, (4, 4, 2))

    @patch("torch.ops")
    def test_quant_mode_other_passthrough(self, mock_ops):
        raw_scale = torch.randn(4, 8)
        self._setup_mock(mock_ops, scale=raw_scale)
        act = NPUMoEInitRouting_v2(quant_mode=1)
        _, _, _, scale = act._init_routing(
            torch.randn(4, 16),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
            num_experts=4,
            top_k=2,
        )
        self.assertIs(scale, raw_scale)

    @patch("torch.ops")
    def test_expert_tokens_int64(self, mock_ops):
        self._setup_mock(mock_ops)
        act = NPUMoEInitRouting_v2()
        _, _, expert_tokens, _ = act._init_routing(
            torch.randn(4, 16),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
            num_experts=4,
            top_k=2,
        )
        self.assertEqual(expert_tokens.dtype, torch.int64)


# =============================================================================
# NPUMoEInitRouting_Quant
# =============================================================================
class TestNPUMoEInitRoutingQuant(unittest.TestCase):
    def _setup_mock(self, mock_ops, scale=None):
        mock_ops.npu.npu_moe_init_routing_quant.return_value = (
            torch.randn(8, 16),
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            torch.tensor([4, 4, 0, 0]),
            None,
            scale,
        )

    @patch("torch.ops")
    def test_op_kwargs(self, mock_ops):
        self._setup_mock(mock_ops)
        hidden = torch.randn(4, 16)
        topk_ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])

        act = NPUMoEInitRouting_Quant()
        act._init_routing(hidden, topk_ids, num_experts=4)

        args, kwargs = mock_ops.npu.npu_moe_init_routing_quant.call_args
        self.assertIs(args[0], hidden)
        self.assertIs(args[1], topk_ids)
        self.assertEqual(kwargs["active_num"], 8)
        self.assertEqual(kwargs["expert_num"], 4)
        self.assertEqual(kwargs["quant_mode"], 1)
        self.assertFalse(kwargs["expert_tokens_before_capacity_flag"])

    @patch("torch.ops")
    def test_returns_pertoken_scale(self, mock_ops):
        raw_scale = torch.randn(8, 8)
        self._setup_mock(mock_ops, scale=raw_scale)
        act = NPUMoEInitRouting_Quant()
        _, _, _, scale = act._init_routing(
            torch.randn(4, 16),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
            num_experts=4,
        )
        self.assertIs(scale, raw_scale)

    @patch("torch.ops")
    def test_expert_tokens_int64(self, mock_ops):
        self._setup_mock(mock_ops)
        act = NPUMoEInitRouting_Quant()
        _, _, expert_tokens, _ = act._init_routing(
            torch.randn(4, 16),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
            num_experts=4,
        )
        self.assertEqual(expert_tokens.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
