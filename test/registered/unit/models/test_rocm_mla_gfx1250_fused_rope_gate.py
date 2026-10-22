"""The gate that sends gfx1250 MLA decode through the fused rope kernel.

fused_qk_rope_cat_and_cache_mla rotates the whole rope head with one cos/sin
table and writes the KV cache itself. Where any term below fails, the unfused
path applies something the kernel does not (a partial rotary dim, a DSA or DCP
layout, prefill), and the fused result would be wrong without an error.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MIXIN = forward_mla_rocm.DeepseekMLARocmForwardMixin
_ROPE_HEAD_DIM = 64


def _gate(attn, forward_batch):
    _MIXIN.init_mla_rocm_forward(attn)
    return _MIXIN._fuse_rope_for_gfx1250_mla(attn, forward_batch)


def _rope(rotary_dim=_ROPE_HEAD_DIM):
    with mock.patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True):
        return RotaryEmbedding(
            head_size=_ROPE_HEAD_DIM,
            rotary_dim=rotary_dim,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=False,
            dtype=torch.float32,
        )


def _attn(**overrides):
    """An attention module every term accepts, before the caller breaks one."""
    attn = SimpleNamespace(
        current_attention_backend="aiter",
        kv_cache_dtype="fp8_e4m3",
        use_dsa=False,
        dcp_size=1,
        rotary_emb=_rope(),
        qk_rope_head_dim=_ROPE_HEAD_DIM,
    )
    for k, v in overrides.items():
        setattr(attn, k, v)
    return attn


def _batch(is_decode=True):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: is_decode))


class TestGfx1250FusedRopeGate(CustomTestCase):
    def setUp(self):
        # The first term is the platform and short-circuits everything else;
        # patch it so the remaining terms are reachable off gfx1250.
        patcher = mock.patch.object(
            forward_mla_rocm, "_use_gfx1250_fused_mla_rope", True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_takes_the_fused_path_when_every_term_holds(self):
        self.assertTrue(_gate(_attn(), _batch()))

    def test_off_gfx1250_nothing_else_is_consulted(self):
        with mock.patch.object(forward_mla_rocm, "_use_gfx1250_fused_mla_rope", False):
            self.assertFalse(_gate(_attn(), _batch()))

    def test_prefill_keeps_the_unfused_path(self):
        self.assertFalse(_gate(_attn(), _batch(is_decode=False)))

    def test_aiter_and_triton_decode(self):
        for backend in ("aiter", "triton"):
            with self.subTest(backend=backend):
                attn = _attn(current_attention_backend=backend)
                self.assertTrue(_gate(attn, _batch()))

    def test_each_term_alone_turns_it_off(self):
        # The kernel serves only plain aiter or triton MLA decode: no DSA
        # indexer layout, no DCP-sharded KV, and a rope to apply.
        for name, broken in (
            ("current_attention_backend", "fa3"),
            ("use_dsa", True),
            ("dcp_size", 2),
            ("rotary_emb", None),
        ):
            with self.subTest(term=name):
                self.assertFalse(_gate(_attn(**{name: broken}), _batch()))

    def test_kv_cache_dtypes(self):
        """The kernel writes only a plain bf16 or fp8 e4m3 cache."""
        for dtype in ("auto", "bf16", "bfloat16", "fp8_e4m3"):
            with self.subTest(dtype=dtype):
                self.assertTrue(_gate(_attn(kv_cache_dtype=dtype), _batch()))
        for dtype in ("fp8_e5m2", "mxfp8", "nvfp4", "fp4_mx_block16"):
            with self.subTest(dtype=dtype):
                self.assertFalse(_gate(_attn(kv_cache_dtype=dtype), _batch()))

    def test_partial_rotary_dim_keeps_the_unfused_path(self):
        """The kernel rotates the whole rope head; a module that rotates only
        part of it would have the rest rotated too."""
        attn = _attn(rotary_emb=_rope(rotary_dim=_ROPE_HEAD_DIM // 2))
        self.assertFalse(_gate(attn, _batch()))


class TestGfx1250RopeTables(CustomTestCase):
    def test_cos_then_sin_halves_of_the_sglang_cache(self):
        """sglang's cache is [cos | sin] per position; handing the kernel the
        halves swapped rotates every query and key the wrong way."""
        rope = _rope()
        with mock.patch.object(forward_mla_rocm, "_use_gfx1250_fused_mla_rope", True):
            cos, sin = forward_mla_rocm._rope_cos_sin(rope)

        position = 5
        inv_freq = rope._compute_inv_freq(rope.base)
        torch.testing.assert_close(cos[position], (position * inv_freq).cos())
        torch.testing.assert_close(sin[position], (position * inv_freq).sin())


if __name__ == "__main__":
    unittest.main()
