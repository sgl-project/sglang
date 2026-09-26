"""Unit tests for srt/layers/layernorm_sp (Megatron LayerNorm sequence parallelism).

Covers the pure logic that gates SP -- the Qwen3 allowlist, the config guards, and
the prefill-only activation rule -- without launching a server. The collectives and
fused matmul fast-paths need a real TP group and are covered by the e2e test.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.arg_groups.layernorm_sp_hook import validate_layernorm_sp
from sglang.srt.layers import communicator as comm
from sglang.srt.layers import layernorm_sp
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import (
    get_flags,
    get_forward,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def _initialize(*, enable=True, arch="Qwen3ForCausalLM"):
    publish(
        ServerArgs(model_path="dummy", enable_layernorm_sp=enable), role="tokenizer"
    )
    layernorm_sp.initialize_layernorm_sp(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch] if arch else [])
        ),
    )


class TestLayerNormSPGating(CustomTestCase):
    def tearDown(self):
        reset_context()

    def test_initialize_enables_only_for_allowlisted_arch(self):
        _initialize(enable=True, arch="Qwen3ForCausalLM")
        self.assertTrue(layernorm_sp.layernorm_sp_enabled())
        # Flag on but unsupported architecture -> stays off.
        _initialize(enable=True, arch="LlamaForCausalLM")
        self.assertFalse(layernorm_sp.layernorm_sp_enabled())
        # Supported architecture but flag off -> stays off.
        _initialize(enable=False, arch="Qwen3ForCausalLM")
        self.assertFalse(layernorm_sp.layernorm_sp_enabled())

    def test_defaults_off_before_initialization(self):
        # A process that never runs initialize_layernorm_sp must not enable SP.
        self.assertFalse(layernorm_sp.layernorm_sp_enabled())

    def test_runs_sp_only_on_extend(self):
        with get_flags().sp.override(enabled=True):
            self.assertTrue(layernorm_sp.runs_sp(ForwardMode.EXTEND))
            # SP is prefill-only; decode must never engage it.
            self.assertFalse(layernorm_sp.runs_sp(ForwardMode.DECODE))
        # A disabled model never activates, even on EXTEND.
        self.assertFalse(layernorm_sp.runs_sp(ForwardMode.EXTEND))

    def test_runs_sp_ignores_the_active_flag(self):
        """Regression: the exit gather must not key off ``sp_active``.

        ``sp_active`` is written by Python inside the CUDA-graph-captured region,
        so it is stale on graph replay. Callers outside that region (the exit
        gather in LogitsProcessor) recompute with ``runs_sp`` instead; if that ever
        starts consulting the flag, a replayed prefill skips the gather and feeds
        sequence-sharded hidden states to the LM head.
        """
        with get_flags().sp.override(enabled=True):
            get_forward().set("sp_active", False)
            self.assertTrue(layernorm_sp.runs_sp(ForwardMode.EXTEND))
        reset_context()


class TestLayerNormSPValidation(CustomTestCase):
    """``validate_layernorm_sp`` is pure; pass config in directly."""

    VALID = dict(
        architecture="Qwen3ForCausalLM",
        tp_size=2,
        enable_dp_attention=False,
        speculative_algorithm=None,
    )

    def test_valid_config_passes(self):
        validate_layernorm_sp(**self.VALID)  # must not raise

    def test_rejects_unsupported_arch(self):
        with self.assertRaisesRegex(ValueError, "only supported"):
            validate_layernorm_sp(**{**self.VALID, "architecture": "LlamaForCausalLM"})

    def test_rejects_tp_size_one(self):
        with self.assertRaisesRegex(ValueError, "tp_size"):
            validate_layernorm_sp(**{**self.VALID, "tp_size": 1})

    def test_rejects_dp_attention(self):
        with self.assertRaisesRegex(ValueError, "dp-attention"):
            validate_layernorm_sp(**{**self.VALID, "enable_dp_attention": True})

    def test_rejects_speculative(self):
        with self.assertRaisesRegex(ValueError, "speculative"):
            validate_layernorm_sp(**{**self.VALID, "speculative_algorithm": "EAGLE3"})


class TestLayerNormSPActiveFlag(CustomTestCase):
    def tearDown(self):
        reset_context()

    def test_active_flag_is_registered_on_forward_flags(self):
        """``sp_active`` must stay a registered ForwardFlags slot.

        ``set()`` rejects names missing from ``ForwardFlags._DEFAULTS``, so this
        fails if the slot is dropped, and it must also be in ``_GRAPH_VISIBLE``
        because the participant linears read it under CUDA graph capture.
        """
        from sglang.srt.runtime_context import ForwardFlags

        self.assertIn("sp_active", ForwardFlags._DEFAULTS)
        self.assertIn("sp_active", ForwardFlags._GRAPH_VISIBLE)

        self.assertFalse(get_forward().sp_active)  # default
        get_forward().set("sp_active", True)
        self.assertTrue(get_forward().sp_active)
        get_forward().set("sp_active", False)
        self.assertFalse(get_forward().sp_active)


class _Norm:
    def __call__(self, x, residual=None, post_residual_addition=None):
        if residual is None:
            return x * 2
        s = x + residual
        return s * 2, s


class TestSpRegionSteps(CustomTestCase):
    """The communicator opens the SP region at the model's first layer on a
    prefill forward and, inside it, keeps every boundary on this rank's sequence
    shard: add + norm, no move, no collective."""

    def communicator(self, *, first_layer):
        c = comm.LayerCommunicator.__new__(comm.LayerCommunicator)
        c._sp_region = True
        c.layer_scatter_modes = SimpleNamespace(is_first_layer=first_layer)
        c._attn_input_fusions = ()
        c.input_layernorm = _Norm()
        c.post_attention_layernorm = _Norm()
        c.qkv_latent_func = None
        c._context = SimpleNamespace(cache=None)
        c._communicate_simple_fn = MagicMock(side_effect=AssertionError("moved"))
        c._communicate_summable_tensor_pair_fn = MagicMock(
            side_effect=AssertionError("postprocess ran")
        )
        return c

    def run_prepare_attn(self, communicator, mode, hidden, residual):
        batch = SimpleNamespace(forward_mode=mode)
        with (
            get_flags().sp.override(enabled=True),
            get_forward().scoped(sp_active=False),
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ),
            patch.object(
                layernorm_sp, "sp_entry_scatter", side_effect=lambda h: h[:1]
            ) as scatter,
        ):
            out = communicator.prepare_attn(hidden, residual, batch)
            return out, get_forward().sp_active, scatter

    def test_the_first_layer_opens_the_region_on_prefill(self):
        hidden = torch.ones(2, 4)
        (h, r), active, scatter = self.run_prepare_attn(
            self.communicator(first_layer=True), ForwardMode.EXTEND, hidden, None
        )
        self.assertTrue(active)
        scatter.assert_called_once()
        torch.testing.assert_close(h, torch.full((1, 4), 2.0))
        torch.testing.assert_close(r, torch.ones(1, 4))

    def test_decode_leaves_the_region_closed(self):
        communicator = self.communicator(first_layer=True)
        communicator._communicate_simple_fn = MagicMock(
            side_effect=lambda **k: k["hidden_states"]
        )
        (h, _), active, scatter = self.run_prepare_attn(
            communicator, ForwardMode.DECODE, torch.ones(2, 4), None
        )
        self.assertFalse(active)
        scatter.assert_not_called()
        communicator._communicate_simple_fn.assert_called_once()

    def test_a_later_layer_does_not_reopen_the_region(self):
        communicator = self.communicator(first_layer=False)
        communicator._communicate_simple_fn = MagicMock(
            side_effect=lambda **k: k["hidden_states"]
        )
        _, active, scatter = self.run_prepare_attn(
            communicator, ForwardMode.EXTEND, torch.ones(2, 4), torch.ones(2, 4)
        )
        self.assertFalse(active)
        scatter.assert_not_called()

    def test_inside_the_region_each_boundary_stays_on_the_shard(self):
        communicator = self.communicator(first_layer=False)
        hidden, residual = torch.ones(1, 4), torch.full((1, 4), 3.0)
        with (
            get_flags().sp.override(enabled=True),
            get_forward().scoped(sp_active=True),
        ):
            with patch.object(
                comm,
                "get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ):
                h, r = communicator.prepare_attn(
                    hidden,
                    residual.clone(),
                    SimpleNamespace(forward_mode=ForwardMode.EXTEND),
                )
            torch.testing.assert_close(h, torch.full((1, 4), 8.0))
            h, r = communicator.prepare_mlp(hidden, residual.clone(), object())
            torch.testing.assert_close(h, torch.full((1, 4), 8.0))
            torch.testing.assert_close(r, torch.full((1, 4), 4.0))
            out = communicator.postprocess_layer(hidden, residual, object())
            self.assertIs(out[0], hidden)
            self.assertIs(out[1], residual)


if __name__ == "__main__":
    unittest.main()
