"""``_LastRowModel``: last-position logits and KV cache match the full forward."""

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

if _HAS_MLX:
    import mlx.core as mx
    from mlx_lm.models import qwen2
    from mlx_lm.models.cache import make_prompt_cache

    from sglang.srt.hardware_backend.mlx.model_runner import _LastRowModel

    class _SoftcappedModel(qwen2.Model):
        def __call__(self, inputs, cache=None, input_embeddings=None):
            out = self.lm_head(self.model(inputs, cache, input_embeddings))
            return mx.tanh(out / 5.0) * 5.0


def _tiny_qwen2(tie: bool, cls=None):
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
    return (cls or qwen2.Model)(args)


@unittest.skipUnless(_HAS_MLX, "requires mlx + mlx_lm")
class TestLastRowLogits(CustomTestCase):
    def _assert_matches_full_forward(self, model):
        mx.random.seed(0)
        x = mx.random.randint(0, 128, (1, 9))
        cache_full, cache_last = make_prompt_cache(model), make_prompt_cache(model)
        full = model(x, cache=cache_full)
        last = type(model).__call__(
            _LastRowModel(model, model.model), x, cache=cache_last
        )
        mx.eval(full, last)

        self.assertEqual(last.shape, (1, 1, 128))
        self.assertTrue(mx.allclose(full[:, -1, :], last[:, -1, :], atol=1e-5).item())
        for c_full, c_last in zip(cache_full, cache_last):
            self.assertEqual(c_full.offset, c_last.offset)
            self.assertTrue(mx.array_equal(c_full.keys, c_last.keys).item())
            self.assertTrue(mx.array_equal(c_full.values, c_last.values).item())
        return last

    def test_tied_and_untied_heads(self):
        for tie in (False, True):
            with self.subTest(tie=tie):
                self._assert_matches_full_forward(_tiny_qwen2(tie=tie))

    def test_post_head_op_still_applies(self):
        last = self._assert_matches_full_forward(
            _tiny_qwen2(tie=False, cls=_SoftcappedModel)
        )
        self.assertTrue((mx.abs(last) <= 5.0).all().item())


if __name__ == "__main__":
    unittest.main()
