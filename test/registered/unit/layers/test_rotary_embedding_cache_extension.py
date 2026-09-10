"""Unit tests for RoPE cos_sin_cache extension across RotaryEmbedding variants.

`_ensure_cos_sin_cache_length()` grows the cos/sin cache at load time to the
reserved context length (see `reserve_rope_cache` in srt/utils/common.py).
The extension rows must be a faithful continuation of the natively built
cache: the same inv_freq and the same per-variant row scaling (YaRN-family
mscale, LinearScaling time scaling).
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.rotary_embedding.base as rotary_base
import sglang.srt.layers.rotary_embedding.mrope as rotary_mrope
from sglang.srt.layers.rotary_embedding.base import (
    LinearScalingRotaryEmbedding,
    RotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.mrope import YaRNScalingMRotaryEmbedding
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
    DynamicNTKAlphaRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    Llama3RotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.yarn import YaRNScalingRotaryEmbedding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_FAKE_EXEC = SimpleNamespace(deterministic=SimpleNamespace(rl_on_policy_target=None))


@contextmanager
def _default_exec_context():
    """Skip the published runtime-context requirement (CPU unit tests)."""
    with (
        patch.object(rotary_base, "get_exec", lambda: _FAKE_EXEC),
        patch.object(rotary_mrope, "get_exec", lambda: _FAKE_EXEC),
    ):
        yield


def _make_base():
    return RotaryEmbedding(64, 64, 1024, 10000, True, torch.float32)


def _make_yarn():
    return YaRNScalingRotaryEmbedding(64, 64, 1024, 10000, True, 4.0, torch.float32)


def _make_deepseek():
    return DeepseekScalingRotaryEmbedding(
        64, 64, 1024, 10000, True, 4.0, torch.float32, device="cpu"
    )


def _make_yarn_mrope():
    return YaRNScalingMRotaryEmbedding(
        64, 64, 1024, 10000, True, 4.0, torch.float32, mrope_section=[8, 12, 12]
    )


def _make_dynamic_ntk():
    return DynamicNTKScalingRotaryEmbedding(
        64, 64, 1024, 10000, True, 4.0, torch.float32
    )


def _make_dynamic_ntk_alpha():
    return DynamicNTKAlphaRotaryEmbedding(
        64, 64, 1024, 10000, True, 30.0, torch.float32
    )


def _make_linear():
    return LinearScalingRotaryEmbedding(64, 64, 1024, 10000, True, [4.0], torch.float32)


def _make_linear_multi():
    # Blocks: [0, 2048) at scale 2.0, [2048, 3072) at scale 1.0. The last
    # block has scale 1.0 but offset 2048, so extension rows must still be
    # shifted by the block offset.
    return LinearScalingRotaryEmbedding(
        64, 64, 1024, 10000, True, [2.0, 1.0], torch.float32
    )


def _make_llama3():
    return Llama3RotaryEmbedding(
        64, 64, 1024, 10000, True, torch.float32, 8.0, 1.0, 4.0, 2048
    )


def _ntk_adjusted_base(rope):
    max_len = rope.max_position_embeddings * rope.scaling_factor
    return rope.base * (
        (rope.scaling_factor * max_len / rope.max_position_embeddings)
        - (rope.scaling_factor - 1)
    ) ** (rope.rotary_dim / (rope.rotary_dim - 2))


def _alpha_adjusted_base(rope):
    return rope.base * rope.scaling_alpha ** (rope.rotary_dim / (rope.rotary_dim - 2))


# name, factory, ground-truth inv_freq source, mscale getter, time scaling
_CASES = [
    ("base", _make_base, lambda r: r._compute_inv_freq(r.base), None, (1.0, 0.0)),
    (
        "yarn",
        _make_yarn,
        lambda r: r._compute_inv_freq(r.scaling_factor),
        "mscale",
        (1.0, 0.0),
    ),
    (
        "deepseek_scaling",
        _make_deepseek,
        lambda r: r._compute_inv_freq(r.scaling_factor),
        "mscale",
        (1.0, 0.0),
    ),
    (
        "yarn_mrope",
        _make_yarn_mrope,
        lambda r: r._compute_inv_freq(r.scaling_factor),
        "mscale",
        (1.0, 0.0),
    ),
    (
        "dynamic_ntk",
        _make_dynamic_ntk,
        lambda r: r._compute_inv_freq(_ntk_adjusted_base(r)),
        None,
        (1.0, 0.0),
    ),
    (
        "dynamic_ntk_alpha",
        _make_dynamic_ntk_alpha,
        lambda r: r._compute_inv_freq(_alpha_adjusted_base(r)),
        None,
        (1.0, 0.0),
    ),
    (
        "linear_scaling",
        _make_linear,
        lambda r: r._compute_inv_freq(r.base),
        None,
        (4.0, 0.0),
    ),
    (
        "llama3",
        _make_llama3,
        lambda r: r._compute_inv_freq(r.base),
        None,
        (1.0, 0.0),
    ),
    (
        "linear_scaling_multi_factor",
        _make_linear_multi,
        lambda r: r._compute_inv_freq(r.base),
        None,
        (1.0, 2048.0),
    ),
]

_NEEDED = 4096


def _continuation_rows(
    inv_freq, start, end, mscale=1.0, time_scale=1.0, time_offset=0.0
):
    t = torch.arange(start, end, dtype=torch.float32)
    if time_scale != 1.0 or time_offset != 0.0:
        t = (t - time_offset) / time_scale
    freqs = torch.einsum("i,j->ij", t, inv_freq.float())
    return torch.cat((freqs.cos() * mscale, freqs.sin() * mscale), dim=-1)


class TestRotaryEmbeddingCacheExtension(CustomTestCase):
    def test_extension_rows_continue_native_cache(self):
        for name, make, inv_freq_fn, mscale_attr, (time_scale, time_offset) in _CASES:
            with self.subTest(case=name):
                with _default_exec_context():
                    rope = make()
                    native_len = rope.cos_sin_cache.shape[0]
                    native_cache = rope.cos_sin_cache.clone()
                    rope._ensure_cos_sin_cache_length(_NEEDED)
                    # Ground truth: continue the native formula, i.e. the
                    # subclass-scaled inv_freq and row scaling.
                    mscale = getattr(rope, mscale_attr) if mscale_attr else 1.0
                    gt = _continuation_rows(
                        inv_freq_fn(rope),
                        native_len,
                        rope.cos_sin_cache.shape[0],
                        mscale=mscale,
                        time_scale=time_scale,
                        time_offset=time_offset,
                    )
                new_len = rope.cos_sin_cache.shape[0]
                self.assertGreaterEqual(new_len, _NEEDED)
                # The natively computed rows must stay bit-identical.
                self.assertTrue(
                    torch.equal(rope.cos_sin_cache[:native_len], native_cache),
                    f"{name}: extension modified native rows",
                )
                ext = rope.cos_sin_cache[native_len:]
                self.assertTrue(
                    torch.equal(ext, gt),
                    f"{name}: max_abs_diff={(ext - gt).abs().max().item()}",
                )

    def test_extension_matches_fresh_longer_native_cache(self):
        # For variants whose inv_freq does not depend on the cache length,
        # extending a short-native cache must reproduce the rows of a freshly
        # constructed longer-native cache bit-for-bit.
        with _default_exec_context():
            short = _make_base()
            short._ensure_cos_sin_cache_length(4000)
            long = RotaryEmbedding(64, 64, 4000, 10000, True, torch.float32)
        self.assertTrue(
            torch.equal(short.cos_sin_cache[1024:4000], long.cos_sin_cache[1024:4000])
        )

        with _default_exec_context():
            short = _make_linear()
            short._ensure_cos_sin_cache_length(8000)
            long = LinearScalingRotaryEmbedding(
                64, 64, 2000, 10000, True, [4.0], torch.float32
            )
        self.assertTrue(
            torch.equal(short.cos_sin_cache[4096:8000], long.cos_sin_cache[4096:8000])
        )

    def test_grok_style_override_falls_back_to_historical_math(self):
        # Grok's ScalingRotaryEmbedding overrides _compute_cos_sin_cache
        # without storing the inv_freq it used, and its native rows are
        # unscaled (its mscale multiply is commented out). Extension must
        # not crash and must reproduce the historical base-class math with
        # no row scaling -- in particular NOT the subclass mscale.
        from sglang.srt.models.grok import ScalingRotaryEmbedding

        with _default_exec_context():
            rope = ScalingRotaryEmbedding(64, 64, 1024, 10000, True, 4.0, torch.float32)
            self.assertFalse(hasattr(rope, "inv_freq"))
            assert rope.mscale != 1.0
            native_len = rope.cos_sin_cache.shape[0]
            native_cache = rope.cos_sin_cache.clone()
            rope._ensure_cos_sin_cache_length(_NEEDED)
            new_len = rope.cos_sin_cache.shape[0]
            gt = _continuation_rows(
                rope._compute_inv_freq(rope.base), native_len, new_len
            )
        self.assertGreaterEqual(new_len, _NEEDED)
        self.assertTrue(torch.equal(rope.cos_sin_cache[:native_len], native_cache))
        self.assertTrue(
            torch.equal(rope.cos_sin_cache[native_len:], gt),
            f"max_abs_diff={(rope.cos_sin_cache[native_len:] - gt).abs().max().item()}",
        )

    def test_meta_inv_freq_falls_back_to_base_math(self):
        # A meta/IPC load that never materialized the stored inv_freq buffer
        # must fall back to the historical base-class math without crashing,
        # and must not apply the subclass row scaling.
        with _default_exec_context():
            rope = _make_yarn()
            native_len = rope.cos_sin_cache.shape[0]
            native_cache = rope.cos_sin_cache.clone()
            rope.inv_freq = torch.empty_like(rope.inv_freq, device="meta")
            rope._ensure_cos_sin_cache_length(_NEEDED)
            new_len = rope.cos_sin_cache.shape[0]
            gt = _continuation_rows(
                rope._compute_inv_freq(rope.base), native_len, new_len
            )
        self.assertGreaterEqual(new_len, _NEEDED)
        self.assertTrue(torch.equal(rope.cos_sin_cache[:native_len], native_cache))
        self.assertTrue(
            torch.equal(rope.cos_sin_cache[native_len:], gt),
            f"max_abs_diff={(rope.cos_sin_cache[native_len:] - gt).abs().max().item()}",
        )

    def test_deepseek_npu_totals_extension(self):
        # On NPU, DeepseekScalingRotaryEmbedding also serves
        # cos/sin_cached_total (emb = cat(freqs, freqs), rows * mscale).
        # The extension must grow them with the same native formula.
        import sglang.srt.layers.rotary_embedding.rope_variant as rotary_variant

        with _default_exec_context(), patch.object(rotary_variant, "_is_npu", True):
            rope = _make_deepseek()
            native_len = rope.cos_cached_total.shape[0]
            self.assertEqual(native_len, rope.cos_sin_cache.shape[0])
            rope._ensure_cos_sin_cache_length(6000)
            new_total_len = rope.cos_cached_total.shape[0]
            inv_freq = rope.inv_freq
            mscale = rope.mscale

        # Totals track the generic cache length in lockstep (always covering
        # the requested max_pos), not max_pos itself.
        self.assertEqual(new_total_len, rope.cos_sin_cache.shape[0])
        self.assertGreaterEqual(new_total_len, 6000)
        t = torch.arange(native_len, new_total_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        gt_cos = torch.cos(emb) * mscale
        gt_sin = torch.sin(emb) * mscale
        self.assertTrue(
            torch.equal(rope.cos_cached_total[native_len:], gt_cos),
            "cos_cached_total extension rows != native formula",
        )
        self.assertTrue(
            torch.equal(rope.sin_cached_total[native_len:], gt_sin),
            "sin_cached_total extension rows != native formula",
        )

    def test_inv_freq_is_non_persistent_buffer(self):
        # The stored tensor must be a registered non-persistent buffer:
        # plain attributes never materialize on meta/IPC load paths, and a
        # persistent buffer would change state_dict.
        with _default_exec_context():
            for name, make, _, _, _ in _CASES:
                if name == "llama3":
                    continue
                rope = make()
                self.assertIn(
                    "inv_freq",
                    dict(rope.named_buffers()),
                    f"{name}: inv_freq must be a registered buffer",
                )
                self.assertNotIn(
                    "inv_freq",
                    rope.state_dict(),
                    f"{name}: inv_freq must not be persistent",
                )

    def test_stored_inv_freq_matches_native_formula(self):
        # The stored inv_freq must be the one the subclass actually used for
        # the native cache, not the base-class derivation.
        with _default_exec_context():
            for name, make, inv_freq_fn, _, _ in _CASES:
                if name == "llama3":
                    continue
                rope = make()
                self.assertTrue(
                    torch.equal(rope.inv_freq, inv_freq_fn(rope)),
                    f"{name}: stored inv_freq != native inv_freq",
                )


if __name__ == "__main__":
    unittest.main()
