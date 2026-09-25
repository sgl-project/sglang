"""Unit tests for last-position-only logits in ``MlxModelRunner._forward_lazy_token``.

mlx-lm models apply the logit head to every position of a chunk and the runner
reads one row, so a prefill chunk of T tokens allocates [T, vocab] logits it never
uses.  ``_LastRowModel`` re-runs a model's own ``__call__`` with the trunk output
sliced to its last row.  These tests pin:

- the last-row logits equal the full forward's last row (same trunk, same head);
- the KV cache after the forward is identical, so decode is unchanged;
- ops a model applies after its head (soft-capping, scaling) still run;
- attribute access on the stand-in resolves to the real model and trunk.
"""

from __future__ import annotations

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=4, suite="stage-a-unit-test-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)
_SKIP_REASON = "requires mlx + mlx_lm"

if _HAS_MLX:
    import mlx.core as mx
    from mlx_lm.models import qwen2
    from mlx_lm.models.cache import make_prompt_cache

    from sglang.srt.hardware_backend.mlx.model_runner import _LastRowModel


def _tiny_qwen2(tie: bool):
    args = qwen2.ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=128,
        rope_theta=10000.0,
        tie_word_embeddings=tie,
    )
    return qwen2.Model(args)


if _HAS_MLX:

    class _SoftcappedModel(qwen2.Model):
        """A qwen2 whose ``__call__`` applies a post-head op, as gemma / cohere do."""

        def __call__(self, inputs, cache=None, input_embeddings=None):
            out = self.model(inputs, cache, input_embeddings)
            out = self.lm_head(out)
            return mx.tanh(out / 5.0) * 5.0


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestLastRowLogits(CustomTestCase):
    def _last_row(self, model, x):
        cache = make_prompt_cache(model)
        out = type(model).__call__(_LastRowModel(model, model.model), x, cache=cache)
        return out, cache

    def _full(self, model, x):
        cache = make_prompt_cache(model)
        return model(x, cache=cache), cache

    def _check_equal(self, model):
        mx.random.seed(0)
        x = mx.random.randint(0, 128, (1, 9))
        full, cache_full = self._full(model, x)
        last, cache_last = self._last_row(model, x)
        mx.eval(full, last)

        self.assertEqual(last.shape, (1, 1, 128))
        self.assertEqual(full.shape, (1, 9, 128))
        # float32 tiny model: the head on one row vs nine is the same math.
        self.assertTrue(mx.allclose(full[:, -1, :], last[:, -1, :], atol=1e-5).item())
        for c_full, c_last in zip(cache_full, cache_last):
            self.assertEqual(c_full.offset, c_last.offset)
            self.assertTrue(mx.array_equal(c_full.keys, c_last.keys).item())
            self.assertTrue(mx.array_equal(c_full.values, c_last.values).item())

    def test_untied_head(self):
        self._check_equal(_tiny_qwen2(tie=False))

    def test_tied_head_through_trunk_attribute(self):
        # Tied heads call ``self.model.embed_tokens.as_linear``: resolved via the trunk wrapper.
        self._check_equal(_tiny_qwen2(tie=True))

    def test_post_head_ops_still_apply(self):
        model = _SoftcappedModel(_tiny_qwen2(tie=False).args)
        self._check_equal(model)
        last, _ = self._last_row(model, mx.random.randint(0, 128, (1, 4)))
        self.assertTrue((mx.abs(last) <= 5.0).all().item())

    def test_single_token_chunk_matches_model(self):
        model = _tiny_qwen2(tie=False)
        x = mx.random.randint(0, 128, (1, 1))
        full, _ = self._full(model, x)
        last, _ = self._last_row(model, x)
        self.assertTrue(mx.allclose(full, last, atol=1e-5).item())

    def test_stand_in_forwards_attributes_without_mutating_model(self):
        model = _tiny_qwen2(tie=True)
        trunk = model.model
        stand_in = _LastRowModel(model, trunk)
        self.assertIs(stand_in.args, model.args)
        self.assertIs(stand_in.model.embed_tokens, trunk.embed_tokens)
        self.assertIs(model.model, trunk)


if __name__ == "__main__":
    unittest.main()
