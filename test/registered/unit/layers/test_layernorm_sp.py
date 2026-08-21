"""Unit tests for srt/layers/layernorm_sp (Megatron LayerNorm sequence parallelism).

Covers the pure logic that gates SP -- the Qwen3 allowlist, the config guards, and
the prefill-only activation rule -- without launching a server. The collectives and
fused matmul fast-paths need a real TP group and are covered by the e2e test.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers import layernorm_sp
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

_MODULE = "sglang.srt.layers.layernorm_sp"


def _fake_config(*, enable=True, arch="Qwen3ForCausalLM"):
    """Patch the two config reads behind ``layernorm_sp_enabled``."""
    return patch.multiple(
        _MODULE,
        get_server_args=lambda: SimpleNamespace(enable_layernorm_sp=enable),
        _model_architecture=lambda: arch,
    )


class TestLayerNormSPGating(CustomTestCase):
    def test_enabled_only_for_allowlisted_arch(self):
        with _fake_config(enable=True, arch="Qwen3ForCausalLM"):
            self.assertTrue(layernorm_sp.layernorm_sp_enabled())
        # Flag on but unsupported architecture -> stays off.
        with _fake_config(enable=True, arch="LlamaForCausalLM"):
            self.assertFalse(layernorm_sp.layernorm_sp_enabled())
        # Supported architecture but flag off -> stays off.
        with _fake_config(enable=False, arch="Qwen3ForCausalLM"):
            self.assertFalse(layernorm_sp.layernorm_sp_enabled())

    def test_runs_sp_only_on_extend(self):
        with _fake_config(enable=True):
            self.assertTrue(layernorm_sp.runs_sp(ForwardMode.EXTEND))
            self.assertFalse(layernorm_sp.runs_sp(ForwardMode.DECODE))
            self.assertTrue(
                layernorm_sp.should_activate_sp(
                    SimpleNamespace(forward_mode=ForwardMode.EXTEND)
                )
            )
        # A disabled model never activates, even on EXTEND.
        with _fake_config(enable=False):
            self.assertFalse(layernorm_sp.runs_sp(ForwardMode.EXTEND))

    def test_runs_sp_ignores_the_active_flag(self):
        """Regression: the exit gather must not key off ``sp_active``.

        ``sp_active`` is written by Python inside the CUDA-graph-captured region,
        so it is stale on graph replay. Callers outside that region (the exit
        gather in LogitsProcessor) recompute with ``runs_sp`` instead; if that ever
        starts consulting the flag, a replayed prefill skips the gather and feeds
        sequence-sharded hidden states to the LM head.
        """
        with _fake_config(enable=True):
            layernorm_sp.set_sp_active(False)
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
        layernorm_sp.validate_layernorm_sp(**self.VALID)  # must not raise

    def test_rejects_unsupported_arch(self):
        with self.assertRaisesRegex(ValueError, "only supported"):
            layernorm_sp.validate_layernorm_sp(
                **{**self.VALID, "architecture": "LlamaForCausalLM"}
            )

    def test_rejects_tp_size_one(self):
        with self.assertRaisesRegex(ValueError, "tp_size"):
            layernorm_sp.validate_layernorm_sp(**{**self.VALID, "tp_size": 1})

    def test_rejects_dp_attention(self):
        with self.assertRaisesRegex(ValueError, "dp-attention"):
            layernorm_sp.validate_layernorm_sp(
                **{**self.VALID, "enable_dp_attention": True}
            )

    def test_rejects_speculative(self):
        with self.assertRaisesRegex(ValueError, "speculative"):
            layernorm_sp.validate_layernorm_sp(
                **{**self.VALID, "speculative_algorithm": "EAGLE3"}
            )


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

        self.assertFalse(layernorm_sp.is_sp_active())  # default
        layernorm_sp.set_sp_active(True)
        self.assertTrue(layernorm_sp.is_sp_active())
        layernorm_sp.set_sp_active(False)
        self.assertFalse(layernorm_sp.is_sp_active())


if __name__ == "__main__":
    unittest.main()
