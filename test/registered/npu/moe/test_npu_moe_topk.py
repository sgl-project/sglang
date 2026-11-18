"""
Unit tests for sglang.srt.hardware_backend.npu.moe.topk.
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

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")

# Mock NPU-only and heavy SGLang modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.norm",
    "sgl_kernel_npu.norm.l1_norm",
    "sglang.srt.eplb",
    "sglang.srt.eplb.expert_distribution",
    "sglang.srt.eplb.expert_location_dispatch",
    "sglang.srt.layers",
    "sglang.srt.layers.moe",
    "sglang.srt.layers.moe.topk",
):
    sys.modules.setdefault(_, MagicMock())

from sglang.srt.hardware_backend.npu.moe.topk import (
    _apply_routed_scaling_after_renorm,
    fused_topk_npu,
)

_MODULE = "sglang.srt.hardware_backend.npu.moe.topk"


def _make_config(**kwargs):
    defaults = dict(
        use_grouped_topk=False,
        renormalize=True,
        correction_bias=None,
        scoring_func="softmax",
        top_k=2,
        apply_routed_scaling_factor_on_output=False,
        routed_scaling_factor=None,
        num_fused_shared_experts=0,
        topk_group=1,
        num_expert_group=1,
        torch_native=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _op_return():
    """3-tuple matching the op unpack pattern."""
    return (
        torch.randn(4, 2),
        torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        torch.tensor(0),
    )


# =============================================================================
# _apply_routed_scaling_after_renorm  (pure logic)
# =============================================================================
class TestApplyRoutedScalingAfterRenorm(unittest.TestCase):
    def test_all_conditions_true_scales(self):
        weights = torch.tensor([1.0, 2.0])
        config = _make_config(
            renormalize=True,
            apply_routed_scaling_factor_on_output=True,
            routed_scaling_factor=2.0,
        )
        out = _apply_routed_scaling_after_renorm(weights, config)
        self.assertTrue(torch.equal(out, weights * 2.0))

    def test_renormalize_false_unchanged(self):
        weights = torch.tensor([1.0, 2.0])
        config = _make_config(
            renormalize=False,
            apply_routed_scaling_factor_on_output=True,
            routed_scaling_factor=2.0,
        )
        out = _apply_routed_scaling_after_renorm(weights, config)
        self.assertIs(out, weights)

    def test_apply_routed_false_unchanged(self):
        weights = torch.tensor([1.0, 2.0])
        config = _make_config(
            renormalize=True,
            apply_routed_scaling_factor_on_output=False,
            routed_scaling_factor=2.0,
        )
        out = _apply_routed_scaling_after_renorm(weights, config)
        self.assertIs(out, weights)

    def test_scaling_factor_none_unchanged(self):
        weights = torch.tensor([1.0, 2.0])
        config = _make_config(
            renormalize=True,
            apply_routed_scaling_factor_on_output=True,
            routed_scaling_factor=None,
        )
        out = _apply_routed_scaling_after_renorm(weights, config)
        self.assertIs(out, weights)


# =============================================================================
# fused_topk_npu — branch selection
# =============================================================================
class TestFusedTopkNpuBranches(unittest.TestCase):
    @patch(f"{_MODULE}.StandardTopKOutput")
    @patch(f"{_MODULE}.capture_routed_experts_if_allowed")
    @patch(f"{_MODULE}.get_global_expert_distribution_recorder")
    @patch("torch.ops")
    def test_sqrtsoftplus_branch(
        self, mock_ops, mock_recorder, mock_capture, mock_output
    ):
        mock_ops.custom.npu_moe_gating_top_k.return_value = _op_return()
        config = _make_config(
            scoring_func="sqrtsoftplus",
            apply_routed_scaling_factor_on_output=True,
            routed_scaling_factor=2.0,
        )
        fused_topk_npu(torch.randn(4, 8), torch.randn(4, 6), config)
        mock_ops.custom.npu_moe_gating_top_k.assert_called_once()

    @patch(f"{_MODULE}.StandardTopKOutput")
    @patch(f"{_MODULE}.capture_routed_experts_if_allowed")
    @patch(f"{_MODULE}.get_global_expert_distribution_recorder")
    @patch(f"{_MODULE}.l1_norm")
    @patch("torch.ops")
    def test_fast_path_calls_l1_norm_when_renormalize(
        self, mock_ops, mock_l1, mock_recorder, mock_capture, mock_output
    ):
        mock_ops.npu.npu_moe_gating_top_k_softmax.return_value = _op_return()
        mock_l1.side_effect = lambda x: x
        config = _make_config(
            scoring_func="softmax",
            use_grouped_topk=False,
            correction_bias=None,
            renormalize=True,
        )
        fused_topk_npu(torch.randn(4, 8), torch.randn(4, 6), config)
        mock_l1.assert_called_once()

    @patch(f"{_MODULE}.StandardTopKOutput")
    @patch(f"{_MODULE}.capture_routed_experts_if_allowed")
    @patch(f"{_MODULE}.get_global_expert_distribution_recorder")
    @patch(f"{_MODULE}.l1_norm")
    @patch("torch.ops")
    def test_fast_path_no_l1_norm_when_not_renormalize(
        self, mock_ops, mock_l1, mock_recorder, mock_capture, mock_output
    ):
        mock_ops.npu.npu_moe_gating_top_k_softmax.return_value = _op_return()
        config = _make_config(
            scoring_func="softmax",
            use_grouped_topk=False,
            correction_bias=None,
            renormalize=False,
        )
        fused_topk_npu(torch.randn(4, 8), torch.randn(4, 6), config)
        mock_l1.assert_not_called()

    @patch(f"{_MODULE}.StandardTopKOutput")
    @patch(f"{_MODULE}.capture_routed_experts_if_allowed")
    @patch(f"{_MODULE}.get_global_expert_distribution_recorder")
    @patch(f"{_MODULE}.l1_norm")
    @patch("torch.ops")
    def test_fast_path_shared_experts_slices_weights(
        self, mock_ops, mock_l1, mock_recorder, mock_capture, mock_output
    ):
        mock_ops.npu.npu_moe_gating_top_k_softmax.return_value = _op_return()
        mock_l1.side_effect = lambda x: x
        config = _make_config(
            scoring_func="softmax",
            use_grouped_topk=False,
            correction_bias=None,
            renormalize=True,
            num_fused_shared_experts=1,
        )
        weights = _op_return()[0]
        fused_topk_npu(torch.randn(4, 8), torch.randn(4, 6), config)
        called_arg = mock_l1.call_args.args[0]
        self.assertEqual(called_arg.shape[-1], weights.shape[-1] - 1)

    @patch(f"{_MODULE}.StandardTopKOutput")
    @patch(f"{_MODULE}.capture_routed_experts_if_allowed")
    @patch(f"{_MODULE}.get_global_expert_distribution_recorder")
    @patch("torch.ops")
    def test_sigmoid_branch_calls_npu_op(
        self, mock_ops, mock_recorder, mock_capture, mock_output
    ):
        mock_ops.npu.npu_moe_gating_top_k.return_value = _op_return()
        # use_grouped_topk=True to skip the fast-path (branch 2)
        config = _make_config(scoring_func="sigmoid", use_grouped_topk=True)
        fused_topk_npu(torch.randn(4, 8), torch.randn(4, 6), config)
        mock_ops.npu.npu_moe_gating_top_k.assert_called_once()

    @patch(f"{_MODULE}.StandardTopKOutput")
    @patch(f"{_MODULE}.capture_routed_experts_if_allowed")
    @patch(f"{_MODULE}.get_global_expert_distribution_recorder")
    @patch("torch.ops")
    def test_correction_bias_branch(
        self, mock_ops, mock_recorder, mock_capture, mock_output
    ):
        mock_ops.npu.npu_moe_gating_top_k.return_value = _op_return()
        bias = torch.randn(6)
        config = _make_config(correction_bias=bias)
        fused_topk_npu(torch.randn(4, 8), torch.randn(4, 6), config)
        kwargs = mock_ops.npu.npu_moe_gating_top_k.call_args.kwargs
        self.assertTrue(torch.equal(kwargs["bias"], bias.to(torch.float32)))

    @patch(f"{_MODULE}.select_experts")
    def test_fallback_branch_calls_select_experts(self, mock_select):
        mock_select.return_value = MagicMock()
        config = _make_config(
            use_grouped_topk=True,
            correction_bias=None,
            scoring_func="softmax",
            renormalize=False,
        )
        hidden = torch.randn(4, 8)
        logits = torch.randn(4, 6)
        fused_topk_npu(hidden, logits, config)
        mock_select.assert_called_once()
        self.assertTrue(config.torch_native)


# =============================================================================
# fused_topk_npu — post-processing
# =============================================================================
class TestFusedTopkNpuPostProcessing(unittest.TestCase):
    @patch(f"{_MODULE}.StandardTopKOutput")
    @patch(f"{_MODULE}.capture_routed_experts_if_allowed")
    @patch(f"{_MODULE}.get_global_expert_distribution_recorder")
    @patch(f"{_MODULE}.topk_ids_logical_to_physical")
    @patch("torch.ops")
    def test_expert_dispatch_info_present_calls_logical_to_physical(
        self, mock_ops, mock_l2p, mock_recorder, mock_capture, mock_output
    ):
        mock_ops.npu.npu_moe_gating_top_k_softmax.return_value = _op_return()
        dispatch_info = MagicMock()
        config = _make_config(scoring_func="softmax")
        fused_topk_npu(
            torch.randn(4, 8),
            torch.randn(4, 6),
            config,
            expert_location_dispatch_info=dispatch_info,
        )
        mock_l2p.assert_called_once()

    @patch(f"{_MODULE}.StandardTopKOutput")
    @patch(f"{_MODULE}.capture_routed_experts_if_allowed")
    @patch(f"{_MODULE}.get_global_expert_distribution_recorder")
    @patch(f"{_MODULE}.topk_ids_logical_to_physical")
    @patch("torch.ops")
    def test_expert_dispatch_info_absent_skips_logical_to_physical(
        self, mock_ops, mock_l2p, mock_recorder, mock_capture, mock_output
    ):
        mock_ops.npu.npu_moe_gating_top_k_softmax.return_value = _op_return()
        config = _make_config(scoring_func="softmax")
        fused_topk_npu(torch.randn(4, 8), torch.randn(4, 6), config)
        mock_l2p.assert_not_called()

    @patch(f"{_MODULE}.StandardTopKOutput")
    @patch(f"{_MODULE}.capture_routed_experts_if_allowed")
    @patch(f"{_MODULE}.get_global_expert_distribution_recorder")
    @patch("torch.ops")
    def test_recorder_called(self, mock_ops, mock_recorder, mock_capture, mock_output):
        mock_ops.npu.npu_moe_gating_top_k_softmax.return_value = _op_return()
        recorder_mock = MagicMock()
        mock_recorder.return_value = recorder_mock
        config = _make_config(scoring_func="softmax")
        fused_topk_npu(torch.randn(4, 8), torch.randn(4, 6), config)
        recorder_mock.on_select_experts.assert_called_once()

    @patch(f"{_MODULE}.StandardTopKOutput")
    @patch(f"{_MODULE}.capture_routed_experts_if_allowed")
    @patch(f"{_MODULE}.get_global_expert_distribution_recorder")
    @patch("torch.ops")
    def test_capture_called(self, mock_ops, mock_recorder, mock_capture, mock_output):
        mock_ops.npu.npu_moe_gating_top_k_softmax.return_value = _op_return()
        config = _make_config(scoring_func="softmax")
        layer_id = 5
        fused_topk_npu(torch.randn(4, 8), torch.randn(4, 6), config, layer_id=layer_id)
        mock_capture.assert_called_once()
        call_args = mock_capture.call_args.args
        self.assertEqual(call_args[1], layer_id)

    @patch(f"{_MODULE}.StandardTopKOutput")
    @patch(f"{_MODULE}.capture_routed_experts_if_allowed")
    @patch(f"{_MODULE}.get_global_expert_distribution_recorder")
    @patch("torch.ops")
    def test_returns_standard_topk_output(
        self, mock_ops, mock_recorder, mock_capture, mock_output
    ):
        mock_ops.npu.npu_moe_gating_top_k_softmax.return_value = _op_return()
        expected = MagicMock(name="output")
        mock_output.return_value = expected
        config = _make_config(scoring_func="softmax")
        out = fused_topk_npu(torch.randn(4, 8), torch.randn(4, 6), config)
        self.assertIs(out, expected)
        mock_output.assert_called_once()


if __name__ == "__main__":
    unittest.main()
