"""Verify MiniMaxM3MoE routes through forward_deepep for Ascend FuseEP.

Tests for the bug where ``MiniMaxM3MoE.forward()`` checked only
``is_deepep()`` and omitted ``is_ascend_fuseep()``, causing:
 1. A spurious ``tensor_model_parallel_all_reduce`` on already-combined output.
 2. Shared experts not replicated (``tp_size=1``).
 3. ``ep_size`` / ``top_k`` left unset.

All tests are CPU-only and run in CI without NPU hardware.
"""

from __future__ import annotations

import sys
import types

# Ensure stub for optional C-extension tvm_ffi if not installed on the system
if "tvm_ffi" not in sys.modules:
    tvm_mock = types.ModuleType("tvm_ffi")
    tvm_mock.Object = object
    tvm_mock.Module = object
    sys.modules["tvm_ffi"] = tvm_mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.runtime_context import get_flags, get_parallel


# ---------------------------------------------------------------------------
# Minimal HF-style config stub matching MiniMax-M3 fields
# ---------------------------------------------------------------------------
def _m3_config(**overrides):
    cfg = SimpleNamespace(
        hidden_size=256,
        hidden_act="silu",
        intermediate_size=64,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        n_shared_experts=1,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        use_routing_bias=False,
        swiglu_alpha=None,
        swiglu_limit=None,
        head_dim=32,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestMiniMaxM3MoEAscendFuseEPDispatch(unittest.TestCase):
    """Ensures ascend_fuseep is recognised as an EP-spanning backend."""

    # ------------------------------------------------------------------ #
    #  Test: forward() dispatches to forward_deepep under ascend_fuseep   #
    # ------------------------------------------------------------------ #
    def test_forward_selects_deepep_path_for_ascend_fuseep(self):
        """Under ``ascend_fuseep`` the MoE layer must call ``forward_deepep``,
        NOT ``forward_normal`` (which would run a redundant TP all-reduce)."""
        import sglang.srt.models.minimax_m3 as m3_mod

        with (
            get_flags().moe.override(a2a_backend=MoeA2ABackend.ASCEND_FUSEEP),
            get_parallel().override(
                tp_size=8,
                tp_rank=0,
                moe_ep_size=8,
            ),
        ):
            moe = self._build_moe_stub(m3_mod)

            # Patch both branches so we can detect which one is called
            moe.forward_deepep = MagicMock(return_value="deepep_result")
            moe.forward_normal = MagicMock(return_value="normal_result")

            dummy_hidden = torch.zeros(4, 256)
            dummy_batch = MagicMock()

            result = moe.forward(dummy_hidden, dummy_batch)

            moe.forward_deepep.assert_called_once()
            moe.forward_normal.assert_not_called()
            self.assertEqual(result, "deepep_result")

    def test_forward_selects_deepep_path_for_deepep(self):
        """Sanity check: ``deepep`` still routes to ``forward_deepep``."""
        import sglang.srt.models.minimax_m3 as m3_mod

        with (
            get_flags().moe.override(a2a_backend=MoeA2ABackend.DEEPEP),
            get_parallel().override(
                tp_size=8,
                tp_rank=0,
                moe_ep_size=8,
            ),
        ):
            moe = self._build_moe_stub(m3_mod)

            moe.forward_deepep = MagicMock(return_value="deepep_result")
            moe.forward_normal = MagicMock(return_value="normal_result")

            dummy_hidden = torch.zeros(4, 256)
            dummy_batch = MagicMock()

            result = moe.forward(dummy_hidden, dummy_batch)

            moe.forward_deepep.assert_called_once()
            moe.forward_normal.assert_not_called()
            self.assertEqual(result, "deepep_result")

    def test_forward_selects_normal_path_for_none_backend(self):
        """Sanity check: ``none`` backend routes to ``forward_normal``."""
        import sglang.srt.models.minimax_m3 as m3_mod

        with (
            get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                moe_ep_size=1,
            ),
        ):
            moe = self._build_moe_stub(m3_mod)

            moe.forward_deepep = MagicMock(return_value="deepep_result")
            moe.forward_normal = MagicMock(return_value="normal_result")

            dummy_hidden = torch.zeros(4, 256)
            dummy_batch = MagicMock()

            result = moe.forward(dummy_hidden, dummy_batch)

            moe.forward_normal.assert_called_once()
            moe.forward_deepep.assert_not_called()
            self.assertEqual(result, "normal_result")

    # ------------------------------------------------------------------ #
    #  Test: __init__ sets ep_size / top_k under ascend_fuseep            #
    # ------------------------------------------------------------------ #
    def test_ep_size_and_top_k_set_for_ascend_fuseep(self):
        """``self.ep_size`` and ``self.top_k`` must be initialised for
        ``ascend_fuseep``, matching the ``deepep`` behaviour."""
        import sglang.srt.models.minimax_m3 as m3_mod

        with (
            get_flags().moe.override(a2a_backend=MoeA2ABackend.ASCEND_FUSEEP),
            get_parallel().override(
                tp_size=8,
                tp_rank=0,
                moe_ep_size=8,
            ),
        ):
            moe = self._build_moe_stub(m3_mod)

            self.assertTrue(
                hasattr(moe, "ep_size"), "ep_size not set for ascend_fuseep"
            )
            self.assertEqual(moe.ep_size, 8)
            self.assertTrue(hasattr(moe, "top_k"), "top_k not set for ascend_fuseep")
            self.assertEqual(moe.top_k, 2)

    def test_ep_size_and_top_k_set_for_deepep(self):
        """Sanity check: ``deepep`` still sets ``ep_size`` / ``top_k``."""
        import sglang.srt.models.minimax_m3 as m3_mod

        with (
            get_flags().moe.override(a2a_backend=MoeA2ABackend.DEEPEP),
            get_parallel().override(
                tp_size=8,
                tp_rank=0,
                moe_ep_size=8,
            ),
        ):
            moe = self._build_moe_stub(m3_mod)

            self.assertTrue(hasattr(moe, "ep_size"))
            self.assertEqual(moe.ep_size, 8)
            self.assertTrue(hasattr(moe, "top_k"))
            self.assertEqual(moe.top_k, 2)

    # ------------------------------------------------------------------ #
    #  Test: shared experts replicated (tp_size=1) under ascend_fuseep    #
    # ------------------------------------------------------------------ #
    def test_shared_experts_tp1_for_ascend_fuseep(self):
        """Shared experts must be replicated (``tp_size=1``) under
        ``ascend_fuseep`` because the routed output is already globally
        combined by the fused kernel."""
        import sglang.srt.models.minimax_m3 as m3_mod

        captured_kwargs = {}

        def _capture_mlp(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        with (
            get_flags().moe.override(a2a_backend=MoeA2ABackend.ASCEND_FUSEEP),
            get_parallel().override(
                tp_size=8,
                tp_rank=0,
                moe_ep_size=8,
            ),
            patch.object(m3_mod, "MiniMaxM3MLP", side_effect=_capture_mlp),
        ):
            self._build_moe_stub(m3_mod)

        self.assertIn("tp_size", captured_kwargs)
        self.assertEqual(captured_kwargs["tp_size"], 1)
        self.assertIn("tp_rank", captured_kwargs)
        self.assertEqual(captured_kwargs["tp_rank"], 0)

    # ------------------------------------------------------------------ #
    #  Helper: build a stub MiniMaxM3MoE without real GPU weights         #
    # ------------------------------------------------------------------ #
    def _build_moe_stub(self, m3_mod):
        """Construct ``MiniMaxM3MoE`` with all heavy dependencies mocked."""
        mock_experts = MagicMock()
        mock_experts.should_fuse_routed_scaling_factor_in_topk = False
        mock_experts.num_local_experts = 8

        with (
            patch.object(
                m3_mod,
                "get_moe_impl_class",
                return_value=lambda **kwargs: mock_experts,
            ),
            patch.object(
                m3_mod,
                "is_shared_experts_fusion_disabled",
                return_value=True,
            ),
            patch.object(
                m3_mod,
                "TopK",
                return_value=MagicMock(),
            ),
            patch.object(
                m3_mod,
                "ReplicatedLinear",
                return_value=MagicMock(),
            ),
            patch.object(
                m3_mod,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(ep_num_redundant_experts=0)
                ),
            ),
        ):
            moe = m3_mod.MiniMaxM3MoE(
                config=_m3_config(),
                layer_id=0,
                quant_config=None,
            )
        return moe


if __name__ == "__main__":
    unittest.main()
