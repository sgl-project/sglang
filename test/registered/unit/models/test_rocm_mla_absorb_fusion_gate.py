"""The gates that route ROCm MLA through fused RoPE + KV-write kernels.

Two predicates: the gfx95 one that also folds in the q absorb, and the gfx1250
decode one for fused_qk_rope_cat_and_cache_mla. Every term names a branch the
unfused path would otherwise have taken, so the fused path is only equivalent
where all of them hold. A term quietly dropped would route a shape the kernel
does not implement, and the wrong answer would look like a plausible one.

The gfx95 caller carries one further term, ``not q_replicate_active``, which
needs a ForwardBatch to evaluate and so is not part of that predicate.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.rotary_embedding.bailing_mrope import BailingMRotaryEmbedding
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_CAN_FUSE = forward_mla_rocm._can_fuse_bmm_rope_cat_and_cache
_ABSORB_CONFIG = forward_mla_rocm._absorb_bmm_config


def _attn(**overrides):
    """An attention module every term accepts, before the caller breaks one."""
    attn = SimpleNamespace(
        w_kc=SimpleNamespace(dtype=torch.float8_e4m3fn),
        rotary_emb=object(),
        current_attention_backend="dsa",
        use_deep_gemm_bmm=False,
        dcp_size=1,
        kv_b_proj=SimpleNamespace(set_lora=False),
        _skip_rope_for_dsa_tilelang_fused=lambda: True,
    )
    for k, v in overrides.items():
        setattr(attn, k, v)
    return attn


class TestFusedAbsorbGate(CustomTestCase):
    def setUp(self):
        # The first term is the platform, and it short-circuits everything else;
        # patch it so the remaining terms are reachable off gfx95.
        self._saved = forward_mla_rocm._use_aiter_gfx95
        forward_mla_rocm._use_aiter_gfx95 = True
        self.addCleanup(setattr, forward_mla_rocm, "_use_aiter_gfx95", self._saved)

    def test_dcp_turns_it_off(self):
        """DCP decode all-gathers q_nope_out, which this path never produces.

        Without this term the fused branch sets q_nope_out to None and
        all_gather_q_for_mla_decode dereferences it.
        """
        self.assertFalse(_CAN_FUSE(_attn(dcp_size=2)))

    def test_takes_the_fused_path_when_every_term_holds(self):
        self.assertTrue(_CAN_FUSE(_attn()))

    def test_off_gfx95_nothing_else_is_consulted(self):
        forward_mla_rocm._use_aiter_gfx95 = False
        # An attn that would otherwise qualify, so only the platform can decide.
        self.assertFalse(_CAN_FUSE(_attn()))

    def test_each_term_alone_turns_it_off(self):
        # The kernel absorbs an fp8 weight, needs a rope to fuse, writes the
        # tilelang DSA cache layout, and has no LoRA or deep-gemm variant.
        for name, broken in (
            ("w_kc", SimpleNamespace(dtype=torch.bfloat16)),
            ("rotary_emb", None),
            ("current_attention_backend", "triton"),
            ("use_deep_gemm_bmm", True),
            ("kv_b_proj", SimpleNamespace(set_lora=True)),
            ("_skip_rope_for_dsa_tilelang_fused", lambda: False),
        ):
            with self.subTest(term=name):
                self.assertFalse(_CAN_FUSE(_attn(**{name: broken})))


class TestAbsorbBmmConfig(CustomTestCase):
    def setUp(self):
        self._saved = forward_mla_rocm._absorb_bmm_num_cu
        forward_mla_rocm._absorb_bmm_num_cu = 256
        self.addCleanup(
            setattr,
            forward_mla_rocm,
            "_absorb_bmm_num_cu",
            self._saved,
        )

    def test_decode_tiles(self):
        cases = (
            ((16, 1, 512), (16, 32, 1)),
            ((16, 48, 512), (16, 128, 1)),
            ((16, 200, 512), (32, 128, 2)),
        )
        for args, expected in cases:
            with self.subTest(args=args):
                config = _ABSORB_CONFIG(*args)
                self.assertEqual(
                    (
                        config["BLOCK_SIZE_M"],
                        config["BLOCK_SIZE_N"],
                        config["waves_per_eu"],
                    ),
                    expected,
                )

    def test_large_row_count_falls_back(self):
        self.assertIsNone(_ABSORB_CONFIG(16, 0, 512))
        self.assertIsNone(_ABSORB_CONFIG(16, 257, 512))


_ROPE_HEAD_DIM = 64
_LAYER_ID = 3
_FP8 = forward_mla_rocm.fp8_dtype
_SENTINEL = 7.0


class _CoreDone(Exception):
    pass


class _Mqa:
    """attn_mqa stand-in that stops core once the rope / KV-write choice is made."""

    def __init__(self):
        self.layer_id = _LAYER_ID
        self.k_scale = torch.tensor(0.5)
        self.save_kv_cache = None

    def __call__(self, q, k, v, forward_batch, *, save_kv_cache):
        self.save_kv_cache = save_kv_cache
        raise _CoreDone


def _rope(rope_cls=RotaryEmbedding, **kwargs):
    with mock.patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True):
        return rope_cls(
            head_size=_ROPE_HEAD_DIM,
            rotary_dim=_ROPE_HEAD_DIM,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=False,
            dtype=torch.float32,
            **kwargs,
        )


class _RocmAttn(forward_mla_rocm.DeepseekMLARocmForwardMixin):
    """An attention module every gfx1250 term accepts, before a case breaks one."""

    def __init__(self, **overrides):
        self.current_attention_backend = "aiter"
        self.kv_cache_dtype = "auto"
        self.use_dsa = False
        self.dcp_size = 1
        self.rotary_emb = _rope()
        self.attn_mqa = _Mqa()
        self._fuse_rope_for_trtllm_mla = lambda forward_batch: False
        vars(self).update(overrides)


def _pool(key):
    layers = {_LAYER_ID: SimpleNamespace(key=lambda: key)}
    return SimpleNamespace(
        dtype=key.dtype, layer_router=SimpleNamespace(read=layers.__getitem__)
    )


def _batch(mode=ForwardMode.DECODE, **fields):
    return SimpleNamespace(forward_mode=mode, **fields)


class _Gfx1250Case(CustomTestCase):
    def setUp(self):
        # The first term is the platform and short-circuits everything else;
        # patch it so the remaining terms are reachable off gfx1250.
        self.patch_attr(forward_mla_rocm, "_use_gfx1250_fused_mla_rope", True)
        self.use_pool(SimpleNamespace(dtype=torch.bfloat16))

    def patch_attr(self, target, name, value):
        patcher = mock.patch.object(target, name, value, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def use_pool(self, key):
        self.patch_attr(forward_mla_rocm, "get_token_to_kv_pool", lambda: _pool(key))

    def fuses(self, attn, forward_batch=None):
        return attn._fuse_rope_for_gfx1250_mla(forward_batch or _batch())


class TestGfx1250FusedRopeGate(_Gfx1250Case):
    def test_takes_the_fused_path_when_every_term_holds(self):
        self.assertTrue(self.fuses(_RocmAttn()))

    def test_off_gfx1250_reads_no_kv_pool(self):
        """Every ROCm decode evaluates the gate; off gfx1250 it must not need
        a KV pool."""
        self.patch_attr(forward_mla_rocm, "_use_gfx1250_fused_mla_rope", False)
        self.patch_attr(forward_mla_rocm, "get_token_to_kv_pool", None)
        self.assertFalse(self.fuses(_RocmAttn()))

    def test_only_decode_fuses(self):
        # Verify and draft-extend carry several tokens per request and their own
        # cache-write rules; the kernel is wired for plain decode only.
        for mode in ForwardMode:
            with self.subTest(mode=mode.name):
                self.assertEqual(
                    self.fuses(_RocmAttn(), _batch(mode)), mode == ForwardMode.DECODE
                )

    def test_triton_decode(self):
        self.assertTrue(self.fuses(_RocmAttn(current_attention_backend="triton")))

    def test_each_term_alone_turns_it_off(self):
        # The kernel serves only plain aiter or triton MLA decode: no DSA
        # indexer layout and no DCP-sharded KV.
        for name, broken in (
            ("current_attention_backend", "fa3"),
            ("use_dsa", True),
            ("dcp_size", 2),
        ):
            with self.subTest(term=name):
                self.assertFalse(self.fuses(_RocmAttn(**{name: broken})))

    def test_only_plain_rope_types_fuse(self):
        """The kernel applies a plain rope from cos_sin_cache. Subclasses such as
        the video mrope Bailing MoE v3 swaps in after the base __init__ compute
        theirs differently, so fusing them applies the wrong rope."""
        cases = (
            (_rope(), True),
            (
                _rope(DeepseekScalingRotaryEmbedding, scaling_factor=4.0, device="cpu"),
                True,
            ),
            (None, False),
            (
                _rope(
                    BailingMRotaryEmbedding, mrope_section=[8, 12, 12], video_rope=True
                ),
                False,
            ),
        )
        for rope, fused in cases:
            with self.subTest(rope=type(rope).__name__):
                self.assertEqual(self.fuses(_RocmAttn(rotary_emb=rope)), fused)

    def test_kv_pool_dtypes(self):
        """The kernel asserts a bf16 or e4m3 cache; an fp16 pool (--dtype float16
        with an auto KV cache) keeps the unfused path."""
        for pool_dtype, fused in (
            (torch.bfloat16, True),
            (_FP8, True),
            (torch.float16, False),
        ):
            with self.subTest(pool=pool_dtype):
                self.use_pool(SimpleNamespace(dtype=pool_dtype))
                self.assertEqual(self.fuses(_RocmAttn()), fused)


class TestGfx1250RopePlacement(_Gfx1250Case):
    def run_core(self, attn, forward_batch):
        """Returns whether core took the fused kernel."""
        calls = []

        def kernel(q_nope, q_pe, k_nope, k_pe, *args, q_out_dtype):
            calls.append(args)
            return torch.cat([q_nope, q_pe], -1), None, None, k_nope

        self.patch_attr(forward_mla_rocm, "fused_qk_rope_cat_and_cache_mla", kernel)
        tokens, heads = 2, 4
        with self.assertRaises(_CoreDone):
            attn.forward_absorb_rocm_core(
                q_pe=torch.zeros(tokens, heads, _ROPE_HEAD_DIM),
                k_pe=torch.zeros(tokens, 1, _ROPE_HEAD_DIM),
                q_nope_out=torch.zeros(tokens, heads, 512),
                k_nope=torch.zeros(tokens, 1, 512),
                forward_batch=forward_batch,
                zero_allocator=None,
                positions=torch.zeros(tokens, dtype=torch.int64),
                topk_indices=None,
                llama_4_scaling=None,
            )
        # The fused kernel writes the cache, so the backend must not write it again.
        self.assertEqual(attn.attn_mqa.save_kv_cache, not calls)
        return bool(calls)

    def test_rope_is_applied_exactly_once(self):
        """Prepare skips rope exactly when core fuses it; a mismatch rotates
        q and k twice or not at all."""
        for mode in ForwardMode:
            with self.subTest(mode=mode.name):
                attn = _RocmAttn()
                forward_batch = _batch(mode, out_cache_loc=torch.zeros(2))
                fused = self.run_core(attn, forward_batch)
                self.assertEqual(fused, mode == ForwardMode.DECODE)
                self.assertNotEqual(attn._applies_rope_in_prepare(forward_batch), fused)


class TestGfx1250FusedRopeCall(_Gfx1250Case):
    def test_kernel_arguments(self):
        """sglang's cache is [cos | sin] per position; swapped halves rotate
        every query and key the wrong way. gfx1250 decode also needs bf16 Q
        even over an fp8 KV cache, and AITER reads slot_mapping with stride 1."""
        kv_cache = torch.zeros(8, 1, 576, dtype=torch.bfloat16)
        self.use_pool(kv_cache)
        attn = _RocmAttn(kv_cache_dtype="fp8_e4m3")
        rope = attn.rotary_emb
        positions = torch.tensor([5])
        storage = torch.tensor([3, 21, 5, 22])
        out_cache_loc = storage[::2]
        calls = []

        def kernel(*args, q_out_dtype):
            args[4].fill_(_SENTINEL)
            calls.append((args, q_out_dtype))
            return args[0], None, None, args[2]

        self.patch_attr(forward_mla_rocm, "fused_qk_rope_cat_and_cache_mla", kernel)
        forward_mla_rocm._fused_rope_cat_and_cache(
            attn,
            torch.empty(1, 16, 512, dtype=torch.bfloat16),
            None,
            None,
            None,
            positions,
            out_cache_loc,
        )

        ((args, q_out_dtype),) = calls
        cache, slot_mapping, kernel_positions, cos, sin, k_scale, is_neox = args[4:]
        self.assertTrue(torch.all(kv_cache == _SENTINEL))
        self.assertTrue(slot_mapping.is_contiguous())
        torch.testing.assert_close(slot_mapping, torch.tensor([3, 5]))
        self.assertIs(kernel_positions, positions)
        self.assertIs(k_scale, attn.attn_mqa.k_scale)
        self.assertIs(is_neox, rope.is_neox_style)
        self.assertEqual(q_out_dtype, torch.bfloat16)
        position = 5
        inv_freq = rope._compute_inv_freq(rope.base)
        for table, expected in ((cos, torch.cos), (sin, torch.sin)):
            self.assertEqual(table.stride(), (rope.rotary_dim, 1))
            torch.testing.assert_close(table[position], expected(position * inv_freq))


if __name__ == "__main__":
    unittest.main()
