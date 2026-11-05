"""The gate that picks the fused q-absorb + RoPE + KV-write kernels.

Every term names a branch that one of the two kernels it replaces would
otherwise have taken, so the fused path is only equivalent where all of them
hold. A term quietly dropped here would route a shape the kernel does not
implement, and the wrong answer would look like a plausible one.

The gate's caller carries one further term, ``not q_replicate_active``,
which is computed in prepare and so is not part of that predicate.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_CAN_FUSE_GATE = forward_mla_rocm._can_fuse_bmm_rope_cat_and_cache
_ABSORB_CONFIG = forward_mla_rocm._absorb_bmm_config


_MXFP4_MAX_TOKENS = forward_mla_rocm._FUSED_FP4_BMM_ROPE_CACHE_MAX_TOKENS


def _CAN_FUSE(attn):
    # FP8 terms ignore the forward mode and token count; the caller gates the mode.
    return _CAN_FUSE_GATE(
        attn, forward_mode=ForwardMode.EXTEND, num_tokens=_MXFP4_MAX_TOKENS + 1
    )


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
        kill_switch = mock.patch.object(
            forward_mla_rocm, "_SGLANG_OPT_HIP_FUSE_MLA_PROJECTION_ROPE_CACHE", True
        )
        kill_switch.start()
        self.addCleanup(kill_switch.stop)

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
        with mock.patch.object(
            forward_mla_rocm, "_SGLANG_OPT_HIP_FUSE_MLA_PROJECTION_ROPE_CACHE", False
        ):
            self.assertFalse(_CAN_FUSE(_attn()))


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


class TestMxfp4ProjectionFusionGate(CustomTestCase):
    """The MXFP4 w_kc branch of ``_can_fuse_bmm_rope_cat_and_cache``."""

    def _make_attn(self):
        # Deliberately no q_lora_rank or v_head_dim: the kernel is key-side only,
        # so a gate that read either would raise AttributeError here.
        return SimpleNamespace(
            current_attention_backend="aiter",
            use_dsa=False,
            use_deep_gemm_bmm=False,
            dcp_size=1,
            kv_b_proj=SimpleNamespace(set_lora=False),
            _skip_rope_for_dsa_tilelang_fused=lambda: False,
            w_kc=torch.empty(1, dtype=torch.uint8),
            w_scale_k=torch.empty(1),
            rotary_emb=object(),
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
        )

    def _make_decode_batch(self):
        return SimpleNamespace(forward_mode=ForwardMode.DECODE)

    @contextlib.contextmanager
    def _aiter_gfx95_env(self):
        with (
            mock.patch.object(forward_mla_rocm, "_use_aiter_gfx95", True),
            mock.patch.object(
                forward_mla_rocm, "_SGLANG_OPT_HIP_FUSE_MLA_PROJECTION_ROPE_CACHE", True
            ),
            get_parallel().override(dcp_enabled=False),
        ):
            yield

    def _can_fuse(self, attn, forward_batch, num_tokens=1):
        return _CAN_FUSE_GATE(
            attn,
            forward_mode=forward_batch.forward_mode,
            num_tokens=num_tokens,
        )

    def test_each_term_alone_turns_it_off(self):
        """Prefill keeps the split path, as do layouts the kernel lacks:
        DSA caches, DCP's all-gathered q_nope_out, non-MXFP4 w_kc, NoPE layers."""
        forward_batch = self._make_decode_batch()
        with self._aiter_gfx95_env():
            self.assertFalse(
                self._can_fuse(
                    self._make_attn(),
                    SimpleNamespace(forward_mode=ForwardMode.EXTEND),
                )
            )
            for name, broken in (
                ("use_dsa", True),
                ("kv_b_proj", SimpleNamespace(set_lora=True)),
                ("w_kc", torch.empty(1, dtype=torch.float8_e4m3fn)),
                ("w_kc", torch.empty(1, dtype=torch.bfloat16)),
                ("w_scale_k", None),
                ("rotary_emb", None),
            ):
                with self.subTest(term=name):
                    attn = self._make_attn()
                    setattr(attn, name, broken)
                    self.assertFalse(self._can_fuse(attn, forward_batch))
            with get_parallel().override(dcp_enabled=True):
                self.assertFalse(self._can_fuse(self._make_attn(), forward_batch))
            with mock.patch.object(
                forward_mla_rocm,
                "_SGLANG_OPT_HIP_FUSE_MLA_PROJECTION_ROPE_CACHE",
                False,
            ):
                self.assertFalse(self._can_fuse(self._make_attn(), forward_batch))

    def test_projection_fusion_gate_targets_aiter_decode(self):
        attn = self._make_attn()
        forward_batch = self._make_decode_batch()

        with self._aiter_gfx95_env():
            self.assertTrue(self._can_fuse(attn, forward_batch))

            attn.current_attention_backend = "flashinfer"
            self.assertFalse(self._can_fuse(attn, forward_batch))

    def test_fuses_up_to_the_measured_token_bound(self):
        """Above the bound the fused grid is slower than the split launches."""
        forward_batch = self._make_decode_batch()
        with self._aiter_gfx95_env():
            self.assertTrue(
                self._can_fuse(
                    self._make_attn(), forward_batch, num_tokens=_MXFP4_MAX_TOKENS
                )
            )
            self.assertFalse(
                self._can_fuse(
                    self._make_attn(), forward_batch, num_tokens=_MXFP4_MAX_TOKENS + 1
                )
            )

    def test_projection_fusion_gate_requires_power_of_two_shapes(self):
        """fused_fp4_bmm_rope_cat_and_cache_mla tiles kv_lora_rank and
        qk_rope_head_dim as unmasked ``tl.arange`` tiles and quantizes the BMM K
        dim in 32-wide MXFP4 groups.  None of that is asserted kernel-side, so a
        merely-divisible-looking shape (384 is a multiple of 128, 96 a multiple
        of 32) must still be rejected here or the kernel reads out of bounds.
        """
        forward_batch = self._make_decode_batch()

        with self._aiter_gfx95_env():
            for field, value in (
                ("kv_lora_rank", 384),
                ("qk_nope_head_dim", 96),
                ("qk_nope_head_dim", 16),
                ("qk_nope_head_dim", 512),
                ("qk_rope_head_dim", 48),
            ):
                with self.subTest(field=field, value=value):
                    attn = self._make_attn()
                    setattr(attn, field, value)
                    self.assertFalse(self._can_fuse(attn, forward_batch))


class TestFusedAbsorbQDtype(CustomTestCase):
    """Gluon MLA decode (bh16bn128) reads bf16 Q at 12 local heads, so with an FP8
    cache the MXFP4 fused kernel must emit bf16 Q there, as the split path does.
    The FP8 w_kc kernel feeds DSA, which keeps FP8 Q at any head count."""

    def _q_out_dtype(self, num_heads, kv_cache_dtype, w_kc_dtype=torch.uint8):
        is_mxfp4 = w_kc_dtype == torch.uint8
        attn = SimpleNamespace(
            kv_cache_dtype=kv_cache_dtype,
            current_attention_backend="aiter" if is_mxfp4 else "dsa",
            w_kc=torch.empty(num_heads, 4, 2, dtype=w_kc_dtype),
            w_scale_k=torch.empty(num_heads, 4, 1, dtype=torch.uint8),
            w_scale=torch.empty(1),
            rotary_emb=SimpleNamespace(
                cos_cache=None, sin_cache=None, is_neox_style=False
            ),
            attn_mqa=SimpleNamespace(layer_id=0, k_scale=None),
        )
        q_nope = torch.empty(3, num_heads, 4, dtype=torch.bfloat16)
        fp4_kernel = mock.Mock()
        fp8_kernel = mock.Mock()
        kv_pool = mock.Mock()
        with (
            mock.patch.object(
                forward_mla_rocm,
                "fused_fp4_bmm_rope_cat_and_cache_mla",
                fp4_kernel,
                create=True,
            ),
            mock.patch.object(
                forward_mla_rocm,
                "fused_fp8_bmm_rope_cat_and_cache_mla",
                fp8_kernel,
                create=True,
            ),
            mock.patch.object(
                forward_mla_rocm, "get_token_to_kv_pool", return_value=kv_pool
            ),
        ):
            forward_mla_rocm._fused_bmm_rope_cat_and_cache(
                attn,
                q_nope=q_nope,
                q_pe=None,
                k_nope=None,
                k_pe=None,
                positions=None,
                out_cache_loc=torch.zeros(3, dtype=torch.int64),
            )
        kernel, unused = (
            (fp4_kernel, fp8_kernel) if is_mxfp4 else (fp8_kernel, fp4_kernel)
        )
        unused.assert_not_called()
        return kernel.call_args.kwargs["q_out_dtype"]

    def test_fp8_cache_with_12_heads_keeps_bf16_q(self):
        self.assertEqual(self._q_out_dtype(12, "fp8_e4m3"), torch.bfloat16)
        self.assertEqual(self._q_out_dtype(8, "fp8_e4m3"), forward_mla_rocm.fp8_dtype)
        self.assertEqual(self._q_out_dtype(12, "auto"), torch.bfloat16)

    def test_fp8_w_kc_keeps_fp8_q_at_12_heads(self):
        self.assertEqual(
            self._q_out_dtype(12, "fp8_e4m3", w_kc_dtype=torch.float8_e4m3fn),
            forward_mla_rocm.fp8_dtype,
        )


if __name__ == "__main__":
    unittest.main()
