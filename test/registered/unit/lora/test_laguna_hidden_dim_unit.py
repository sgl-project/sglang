"""Unit tests for Laguna's per-layer LoRA hidden-dim resolution.

Laguna (`poolside/Laguna-XS.2`) sizes each layer's attention from
`num_attention_heads_per_layer[layer_idx]`, so `config.num_attention_heads` —
a single global value — is wrong for any layer whose head count differs. The
generic `get_default_hidden_dim` fallback would use that global value and
mis-size the `qkv_proj` / `o_proj` LoRA buffers, which crashes at generation
with `sgemm_lora_a.py: assert x.shape[-1] == K`.

`LagunaModel.get_hidden_dim` overrides just the two attention projections with
per-layer widths and delegates every other module to the shared helper. These
tests exercise that method directly against a minimal fake config — no CUDA,
no server, no real weights — so they stay hermetic and fast.

Usage:
    python -m pytest test/registered/unit/lora/test_laguna_hidden_dim_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

# CPU-only unit test; no CUDA/distributed dependencies.
register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest

from sglang.srt.configs.laguna import LagunaConfig
from sglang.srt.lora.utils import get_default_hidden_dim
from sglang.srt.models.laguna import LagunaForCausalLM, LagunaModel


def _make_fake_laguna(num_attention_heads_per_layer):
    """Build a `LagunaModel` stand-in around a real `LagunaConfig`, without
    running `__init__` (no weights, CPU-only).

    `head_dim` (128) is deliberately not `hidden_size // num_attention_heads`,
    matching every real Laguna checkpoint — so the hook must read `head_dim`
    directly, never derive it. `LagunaConfig` sets the global
    `num_attention_heads` from the first full-attention layer.
    """
    config = LagunaConfig(
        hidden_size=2048,
        head_dim=128,
        num_key_value_heads=8,
        num_hidden_layers=len(num_attention_heads_per_layer),
        num_attention_heads_per_layer=list(num_attention_heads_per_layer),
    )
    model = LagunaModel.__new__(LagunaModel)
    model.config = config
    return model


class TestLagunaPerLayerAttentionDims(unittest.TestCase):
    """`qkv_proj` / `o_proj` must follow the *layer's own* head count."""

    def setUp(self):
        # Layer 0: 48 heads (== global default). Layer 1: 64 heads (differs).
        self.model = _make_fake_laguna([48, 64])
        self.head_dim = 128
        self.hidden_size = 2048
        self.num_kv_heads = 8

    def test_qkv_proj_uses_first_layer_head_count(self):
        got = self.model.get_hidden_dim("qkv_proj", layer_idx=0)
        expected_out = self.head_dim * (48 + self.num_kv_heads * 2)
        self.assertEqual(got, (self.hidden_size, expected_out))

    def test_qkv_proj_uses_wider_layer_head_count(self):
        # The layer that the global fallback would size incorrectly.
        got = self.model.get_hidden_dim("qkv_proj", layer_idx=1)
        expected_out = self.head_dim * (64 + self.num_kv_heads * 2)
        self.assertEqual(got, (self.hidden_size, expected_out))

    def test_o_proj_uses_first_layer_head_count(self):
        got = self.model.get_hidden_dim("o_proj", layer_idx=0)
        self.assertEqual(got, (self.head_dim * 48, self.hidden_size))

    def test_o_proj_uses_wider_layer_head_count(self):
        got = self.model.get_hidden_dim("o_proj", layer_idx=1)
        self.assertEqual(got, (self.head_dim * 64, self.hidden_size))

    def test_wider_layer_differs_from_generic_fallback(self):
        """The regression guard: for the wider layer, the model hook must
        return a DIFFERENT (correct) dim than the generic global-head
        fallback — otherwise the buffer is mis-sized and generation asserts
        in `sgemm_lora_a.py`.
        """
        for module_name in ("qkv_proj", "o_proj"):
            hook = self.model.get_hidden_dim(module_name, layer_idx=1)
            generic = get_default_hidden_dim(module_name, self.model.config, 1)
            self.assertNotEqual(
                hook,
                generic,
                f"{module_name}: per-layer hook must diverge from the global "
                "fallback on the wider layer",
            )

    def test_matching_layer_agrees_with_generic_fallback(self):
        """For a layer whose width == the global default, the hook and the
        generic fallback must agree (no gratuitous divergence)."""
        for module_name in ("qkv_proj", "o_proj"):
            hook = self.model.get_hidden_dim(module_name, layer_idx=0)
            generic = get_default_hidden_dim(module_name, self.model.config, 0)
            self.assertEqual(hook, generic)

    def test_missing_head_dim_raises_not_derives(self):
        """Removing the fallback means an absent head_dim fails loudly instead
        of silently deriving a wrong hidden_size // num_attention_heads."""
        cfg = self.model.config
        del cfg.head_dim
        with self.assertRaises(AttributeError):
            self.model.get_hidden_dim("o_proj", layer_idx=1)


class TestLagunaNonAttentionDelegates(unittest.TestCase):
    """Non-attention modules must delegate to the shared helper so that
    `--lora-target-modules all` (MLP / MoE / embed / lm_head) still works
    once the model defines `get_hidden_dim`.
    """

    def setUp(self):
        self.model = _make_fake_laguna([48, 64])

    def test_delegates_mlp_and_embedding_modules(self):
        for module_name in (
            "gate_up_proj",
            "down_proj",
            "gate_up_proj_moe",
            "down_proj_moe",
            "embed_tokens",
            "lm_head",
        ):
            for layer_idx in (0, 1):
                self.assertEqual(
                    self.model.get_hidden_dim(module_name, layer_idx),
                    get_default_hidden_dim(module_name, self.model.config, layer_idx),
                    f"{module_name}@{layer_idx} should match the shared helper",
                )

    def test_unknown_module_raises(self):
        with self.assertRaises(NotImplementedError):
            self.model.get_hidden_dim("not_a_module", layer_idx=0)


class TestLagunaForCausalLMDelegation(unittest.TestCase):
    """`LagunaForCausalLM.get_hidden_dim` must forward to the inner model."""

    def test_forwards_to_inner_model(self):
        inner = _make_fake_laguna([48, 64])
        causal = LagunaForCausalLM.__new__(LagunaForCausalLM)
        # Bypass nn.Module.__setattr__: __new__ skips __init__, so the module
        # registries it expects when assigning a Module-valued attr don't exist.
        object.__setattr__(causal, "model", inner)
        for module_name in ("qkv_proj", "o_proj", "gate_up_proj"):
            for layer_idx in (0, 1):
                self.assertEqual(
                    causal.get_hidden_dim(module_name, layer_idx),
                    inner.get_hidden_dim(module_name, layer_idx),
                )


if __name__ == "__main__":
    unittest.main()
