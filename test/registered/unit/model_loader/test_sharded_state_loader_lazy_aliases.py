"""Regression test for sgl-project/sglang#25332.

DeepSeek-V2 MLA initializes ``self.attn_mha.kv_b_proj`` to ``None`` and
lazily aliases it to the outer ``self.kv_b_proj`` inside ``forward_prepare``.
Saved sharded checkpoints (produced after a warmup forward) encode the
parameter under the ``...attn_mha.kv_b_proj.*`` key, so the load-time
``state_dict`` view must reflect the same aliasing or the keys cannot be
resolved. ``_trigger_lazy_module_aliases`` installs the alias eagerly in
``ShardedStateLoader.load_model`` before reading ``state_dict``; this
module covers that helper.
"""

import unittest

import torch.nn as nn

from sglang.srt.model_loader.loader import _trigger_lazy_module_aliases


def _make_attn_module(out_features: int = 8) -> nn.Module:
    """Mock ``DeepseekV2AttentionMLA``: outer kv_b_proj + attn_mha child
    where ``attn_mha.kv_b_proj`` starts as None (the lazy alias slot)."""
    attn = nn.Module()
    attn.kv_b_proj = nn.Linear(4, out_features, bias=False)
    attn.attn_mha = nn.Module()
    attn.attn_mha.kv_b_proj = None
    return attn


class TestTriggerLazyModuleAliases(unittest.TestCase):
    def test_alias_is_installed_eagerly(self):
        model = nn.Module()
        model.self_attn = _make_attn_module()

        _trigger_lazy_module_aliases(model)

        # attn_mha.kv_b_proj must point at the outer kv_b_proj module
        # (not just an equal copy — same object).
        self.assertIs(
            model.self_attn.attn_mha.kv_b_proj,
            model.self_attn.kv_b_proj,
        )

    def test_state_dict_includes_aliased_key(self):
        """After aliasing, ``model.state_dict()`` must expose the
        ``attn_mha.kv_b_proj.weight`` key the saved checkpoint uses."""
        model = nn.Module()
        model.self_attn = _make_attn_module()

        _trigger_lazy_module_aliases(model)

        keys = set(model.state_dict().keys())
        self.assertIn("self_attn.kv_b_proj.weight", keys)
        self.assertIn("self_attn.attn_mha.kv_b_proj.weight", keys)

        # And both keys must share storage (alias, not copy) so loading the
        # checkpoint into one populates the other.
        outer = model.state_dict()["self_attn.kv_b_proj.weight"]
        inner = model.state_dict()["self_attn.attn_mha.kv_b_proj.weight"]
        self.assertEqual(
            outer.untyped_storage().data_ptr(),
            inner.untyped_storage().data_ptr(),
        )

    def test_idempotent_when_alias_already_installed(self):
        model = nn.Module()
        model.self_attn = _make_attn_module()

        _trigger_lazy_module_aliases(model)
        first_alias = model.self_attn.attn_mha.kv_b_proj

        # Second call must be a no-op, not re-bind to a new object.
        _trigger_lazy_module_aliases(model)
        self.assertIs(model.self_attn.attn_mha.kv_b_proj, first_alias)

    def test_skips_modules_without_attn_mha(self):
        """Modules that don't have an ``attn_mha`` child must be left alone."""
        model = nn.Module()
        model.self_attn = nn.Module()
        model.self_attn.kv_b_proj = nn.Linear(4, 8, bias=False)
        # No attn_mha attribute on self_attn.

        # Must not raise AttributeError or otherwise mutate.
        _trigger_lazy_module_aliases(model)

        self.assertFalse(hasattr(model.self_attn, "attn_mha"))

    def test_skips_modules_without_outer_kv_b_proj(self):
        """If a module has attn_mha but no outer kv_b_proj, leave the
        attn_mha.kv_b_proj slot alone — there's nothing to alias to."""
        model = nn.Module()
        model.some_block = nn.Module()
        model.some_block.attn_mha = nn.Module()
        model.some_block.attn_mha.kv_b_proj = None

        _trigger_lazy_module_aliases(model)

        self.assertIsNone(model.some_block.attn_mha.kv_b_proj)

    def test_does_not_overwrite_non_none_alias(self):
        """If ``attn_mha.kv_b_proj`` is already set to something (e.g. a
        different module on a subclass), don't clobber it."""
        model = nn.Module()
        model.self_attn = _make_attn_module()
        sentinel = nn.Linear(4, 8, bias=False)
        model.self_attn.attn_mha.kv_b_proj = sentinel

        _trigger_lazy_module_aliases(model)

        # Must keep the pre-existing alias, not replace it with the outer
        # kv_b_proj.
        self.assertIs(model.self_attn.attn_mha.kv_b_proj, sentinel)
        self.assertIsNot(model.self_attn.attn_mha.kv_b_proj, model.self_attn.kv_b_proj)


if __name__ == "__main__":
    unittest.main()
