"""
Unit tests for sglang.srt.hardware_backend.npu.moe.finalize_routing.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Mock optional third-party deps pulled in by sglang/__init__ when they are
# not installed, so these tests also run on a CPU-only box (CI installs them).
for _mod in ("triton", "IPython", "IPython.display", "aiohttp", "zmq"):
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

from sglang.srt.hardware_backend.npu.moe.finalize_routing import (
    AllGatherFinalizeRoutingWrapper,
    BaseFinalizeRouting,
    NPUFinalizeRouting,
    NPUMoETokenUnpermute,
)


# =============================================================================
# BaseFinalizeRouting
# =============================================================================
class TestBaseFinalizeRouting(unittest.TestCase):
    def test_is_abstract(self):
        """BaseFinalizeRouting cannot be instantiated without _finalize_routing."""
        with self.assertRaises(TypeError):
            BaseFinalizeRouting()

    def test_abstract_method_exists(self):
        self.assertTrue(hasattr(BaseFinalizeRouting, "_finalize_routing"))


# =============================================================================
# NPUFinalizeRouting
# =============================================================================
class TestNPUFinalizeRoutingInit(unittest.TestCase):
    def test_default_drop_pad_mode(self):
        act = NPUFinalizeRouting()
        self.assertEqual(act.drop_pad_mode, 0)

    def test_custom_drop_pad_mode(self):
        act = NPUFinalizeRouting(drop_pad_mode=1)
        self.assertEqual(act.drop_pad_mode, 1)

    def test_drop_pad_mode_two(self):
        act = NPUFinalizeRouting(drop_pad_mode=2)
        self.assertEqual(act.drop_pad_mode, 2)


class TestNPUFinalizeRoutingApply(unittest.TestCase):
    @patch("torch.ops")
    def test_calls_npu_moe_finalize_routing(self, mock_ops):
        """torch.ops.npu.npu_moe_finalize_routing is called."""
        mock_ops.npu.npu_moe_finalize_routing.return_value = torch.randn(4, 8)

        act = NPUFinalizeRouting()
        act._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )

        mock_ops.npu.npu_moe_finalize_routing.assert_called_once()

    @patch("torch.ops")
    def test_hidden_states_passed_positionally(self, mock_ops):
        mock_ops.npu.npu_moe_finalize_routing.return_value = torch.randn(4, 8)

        hidden = torch.randn(4, 8)
        act = NPUFinalizeRouting()
        act._finalize_routing(
            hidden,
            torch.randn(4, 2),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )

        args, _ = mock_ops.npu.npu_moe_finalize_routing.call_args
        self.assertIs(args[0], hidden)

    @patch("torch.ops")
    def test_kwargs_mapping(self, mock_ops):
        """Parameters are forwarded under the expected keyword names."""
        mock_ops.npu.npu_moe_finalize_routing.return_value = torch.randn(4, 8)

        hidden = torch.randn(4, 8)
        topk_weights = torch.randn(4, 2)
        expanded_row_idx = torch.tensor([0, 1, 2, 3])
        topk_ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])

        act = NPUFinalizeRouting()
        act._finalize_routing(hidden, topk_weights, expanded_row_idx, topk_ids)

        _, kwargs = mock_ops.npu.npu_moe_finalize_routing.call_args
        self.assertIs(kwargs["scales"], topk_weights)
        self.assertIs(kwargs["expanded_src_to_dst_row"], expanded_row_idx)
        self.assertIs(kwargs["export_for_source_row"], topk_ids)

    @patch("torch.ops")
    def test_skip_and_bias_none(self, mock_ops):
        """skip1, skip2, bias are always None."""
        mock_ops.npu.npu_moe_finalize_routing.return_value = torch.randn(4, 8)

        act = NPUFinalizeRouting()
        act._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )

        _, kwargs = mock_ops.npu.npu_moe_finalize_routing.call_args
        self.assertIsNone(kwargs["skip1"])
        self.assertIsNone(kwargs["skip2"])
        self.assertIsNone(kwargs["bias"])

    @patch("torch.ops")
    def test_drop_pad_mode_forwarded(self, mock_ops):
        mock_ops.npu.npu_moe_finalize_routing.return_value = torch.randn(4, 8)

        act = NPUFinalizeRouting(drop_pad_mode=2)
        act._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )

        _, kwargs = mock_ops.npu.npu_moe_finalize_routing.call_args
        self.assertEqual(kwargs["drop_pad_mode"], 2)

    @patch("torch.ops")
    def test_returns_op_output(self, mock_ops):
        expected = torch.randn(4, 8)
        mock_ops.npu.npu_moe_finalize_routing.return_value = expected

        act = NPUFinalizeRouting()
        out = act._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )
        self.assertIs(out, expected)


# =============================================================================
# NPUMoETokenUnpermute
# =============================================================================
class TestNPUMoETokenUnpermute(unittest.TestCase):
    @patch("torch.ops")
    def test_calls_npu_moe_token_unpermute(self, mock_ops):
        mock_ops.npu.npu_moe_token_unpermute.return_value = torch.randn(4, 8)

        act = NPUMoETokenUnpermute()
        act._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )

        mock_ops.npu.npu_moe_token_unpermute.assert_called_once()

    @patch("torch.ops")
    def test_kwargs_mapping(self, mock_ops):
        mock_ops.npu.npu_moe_token_unpermute.return_value = torch.randn(4, 8)

        hidden = torch.randn(4, 8)
        topk_weights = torch.randn(4, 2)
        expanded_row_idx = torch.tensor([0, -1, 2, -3])
        topk_ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])

        act = NPUMoETokenUnpermute()
        act._finalize_routing(hidden, topk_weights, expanded_row_idx, topk_ids)

        _, kwargs = mock_ops.npu.npu_moe_token_unpermute.call_args
        self.assertIs(kwargs["permuted_tokens"], hidden)
        self.assertIs(kwargs["probs"], topk_weights)

    @patch("torch.ops")
    def test_sorted_indices_is_abs(self, mock_ops):
        """sorted_indices is expanded_row_idx.abs()."""
        mock_ops.npu.npu_moe_token_unpermute.return_value = torch.randn(4, 8)

        expanded_row_idx = torch.tensor([0, -1, -2, 3])
        act = NPUMoETokenUnpermute()
        act._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            expanded_row_idx,
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )

        _, kwargs = mock_ops.npu.npu_moe_token_unpermute.call_args
        self.assertTrue(torch.equal(
            kwargs["sorted_indices"], expanded_row_idx.abs()
        ))

    @patch("torch.ops")
    def test_sorted_indices_all_positive(self, mock_ops):
        """Even negative indices become positive after abs()."""
        mock_ops.npu.npu_moe_token_unpermute.return_value = torch.randn(4, 8)

        expanded_row_idx = torch.tensor([-5, -3, -1, 0])
        act = NPUMoETokenUnpermute()
        act._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            expanded_row_idx,
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )

        _, kwargs = mock_ops.npu.npu_moe_token_unpermute.call_args
        self.assertTrue((kwargs["sorted_indices"] >= 0).all())

    @patch("torch.ops")
    def test_returns_op_output(self, mock_ops):
        expected = torch.randn(4, 8)
        mock_ops.npu.npu_moe_token_unpermute.return_value = expected

        act = NPUMoETokenUnpermute()
        out = act._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )
        self.assertIs(out, expected)

    @patch("torch.ops")
    def test_topk_ids_not_used(self, mock_ops):
        """topk_ids is accepted but not forwarded to the unpermute op."""
        mock_ops.npu.npu_moe_token_unpermute.return_value = torch.randn(4, 8)

        act = NPUMoETokenUnpermute()
        act._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )

        _, kwargs = mock_ops.npu.npu_moe_token_unpermute.call_args
        self.assertNotIn("topk_ids", kwargs)
        self.assertNotIn("export_for_source_row", kwargs)


# =============================================================================
# AllGatherFinalizeRoutingWrapper
# =============================================================================
class TestAllGatherFinalizeRoutingWrapperInit(unittest.TestCase):
    def test_defaults(self):
        inner = MagicMock()
        wrapper = AllGatherFinalizeRoutingWrapper(inner)
        self.assertIs(wrapper.inner, inner)
        self.assertEqual(wrapper.dim, -1)

    def test_custom_dim(self):
        inner = MagicMock()
        wrapper = AllGatherFinalizeRoutingWrapper(inner, dim=0)
        self.assertEqual(wrapper.dim, 0)


class TestAllGatherFinalizeRoutingWrapperApply(unittest.TestCase):
    _GET_PARALLEL = (
        "sglang.srt.hardware_backend.npu.moe.finalize_routing.get_parallel"
    )
    _ALL_GATHER = (
        "sglang.srt.hardware_backend.npu.moe.finalize_routing."
        "tensor_model_parallel_all_gather"
    )

    @patch(_ALL_GATHER)
    @patch(_GET_PARALLEL)
    def test_tp1_no_gather(self, mock_get_parallel, mock_all_gather):
        mock_get_parallel.return_value = SimpleNamespace(tp_size=1)
        inner_out = torch.randn(4, 8)
        inner = MagicMock()
        inner._finalize_routing.return_value = inner_out

        wrapper = AllGatherFinalizeRoutingWrapper(inner)
        out = wrapper._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )

        mock_all_gather.assert_not_called()
        self.assertIs(out, inner_out)

    @patch(_ALL_GATHER)
    @patch(_GET_PARALLEL)
    def test_tp2_gathers(self, mock_get_parallel, mock_all_gather):
        mock_get_parallel.return_value = SimpleNamespace(tp_size=2)
        gathered = torch.randn(8, 8)
        mock_all_gather.return_value = gathered
        inner = MagicMock()
        inner._finalize_routing.return_value = torch.randn(4, 8)

        wrapper = AllGatherFinalizeRoutingWrapper(inner, dim=-1)
        out = wrapper._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )

        mock_all_gather.assert_called_once()
        self.assertIs(out, gathered)

    @patch(_ALL_GATHER)
    @patch(_GET_PARALLEL)
    def test_gather_uses_inner_output(self, mock_get_parallel, mock_all_gather):
        """all_gather receives the inner's output, not the raw input."""
        mock_get_parallel.return_value = SimpleNamespace(tp_size=2)
        mock_all_gather.return_value = torch.randn(8, 8)
        inner_out = torch.randn(4, 8)
        inner = MagicMock()
        inner._finalize_routing.return_value = inner_out

        wrapper = AllGatherFinalizeRoutingWrapper(inner, dim=-1)
        wrapper._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )

        args, kwargs = mock_all_gather.call_args
        self.assertIs(args[0], inner_out)
        self.assertEqual(kwargs["dim"], -1)

    @patch(_ALL_GATHER)
    @patch(_GET_PARALLEL)
    def test_gather_with_custom_dim(self, mock_get_parallel, mock_all_gather):
        mock_get_parallel.return_value = SimpleNamespace(tp_size=4)
        mock_all_gather.return_value = torch.randn(16, 8)
        inner = MagicMock()
        inner._finalize_routing.return_value = torch.randn(4, 8)

        wrapper = AllGatherFinalizeRoutingWrapper(inner, dim=0)
        wrapper._finalize_routing(
            torch.randn(4, 8),
            torch.randn(4, 2),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
        )

        _, kwargs = mock_all_gather.call_args
        self.assertEqual(kwargs["dim"], 0)

    @patch(_ALL_GATHER)
    @patch(_GET_PARALLEL)
    def test_forwards_all_args_to_inner(self, mock_get_parallel, mock_all_gather):
        mock_get_parallel.return_value = SimpleNamespace(tp_size=1)
        inner = MagicMock()
        inner._finalize_routing.return_value = torch.randn(4, 8)

        wrapper = AllGatherFinalizeRoutingWrapper(inner)
        hidden = torch.randn(4, 8)
        topk_weights = torch.randn(4, 2)
        expanded_row_idx = torch.tensor([0, 1, 2, 3])
        topk_ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])

        wrapper._finalize_routing(
            hidden, topk_weights, expanded_row_idx, topk_ids
        )

        inner._finalize_routing.assert_called_once_with(
            hidden, topk_weights, expanded_row_idx, topk_ids
        )


if __name__ == "__main__":
    unittest.main()
