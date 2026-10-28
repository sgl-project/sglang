"""The gate that picks the fused q-absorb + RoPE + KV-write kernel.

Every term names a branch that one of the two kernels it replaces would
otherwise have taken, so the fused path is only equivalent where all of them
hold. A term quietly dropped here would route a shape the kernel does not
implement, and the wrong answer would look like a plausible one.

The caller carries one further term, ``not q_replicate_active``, which needs a
ForwardBatch to evaluate and so is not part of this predicate.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

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
        _skip_rope_for_dsa_aiter_fused_prep=lambda: False,
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

    def test_aiter_uses_rope_cache_fusion_without_absorb_fusion(self):
        backend = SimpleNamespace(
            current_attention_backend="dsa",
        )
        backend._skip_rope_for_dsa_aiter_fused_prep = forward_mla_rocm.DeepseekMLARocmForwardMixin._skip_rope_for_dsa_aiter_fused_prep.__get__(
            backend
        )
        saved = forward_mla_rocm.get_exec
        forward_mla_rocm.get_exec = lambda: SimpleNamespace(
            kernel=SimpleNamespace(dsa_decode_backend="aiter")
        )
        self.addCleanup(setattr, forward_mla_rocm, "get_exec", saved)

        self.assertTrue(backend._skip_rope_for_dsa_aiter_fused_prep())
        self.assertFalse(
            _CAN_FUSE(
                _attn(
                    _skip_rope_for_dsa_tilelang_fused=lambda: False,
                    _skip_rope_for_dsa_aiter_fused_prep=lambda: True,
                )
            )
        )

    def test_aiter_dsa_fused_prep_emits_fp8_q(self):
        attn = SimpleNamespace(
            kv_cache_dtype="fp8_e4m3",
            current_attention_backend="dsa",
            attn_mqa=SimpleNamespace(layer_id=0, k_scale=torch.ones(())),
            rotary_emb=SimpleNamespace(
                cos_cache=object(),
                sin_cache=object(),
                is_neox_style=False,
            ),
        )
        kv_pool = SimpleNamespace(get_key_buffer=Mock(return_value=torch.empty(1)))
        fused_op = Mock(
            return_value=(
                torch.empty(1),
                torch.empty(1),
                torch.empty(1),
                torch.empty(1),
            )
        )

        with (
            patch.object(
                forward_mla_rocm,
                "get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(dsa_decode_backend="aiter")
                ),
            ),
            patch.object(
                forward_mla_rocm,
                "get_token_to_kv_pool",
                return_value=kv_pool,
            ),
            patch.object(
                forward_mla_rocm,
                "fused_qk_rope_cat_and_cache_mla",
                fused_op,
            ),
        ):
            forward_mla_rocm._fused_rope_cat_and_cache(
                attn,
                q_nope_out=torch.empty(1, 16, 512, dtype=torch.bfloat16),
                q_pe=torch.empty(1, 16, 64, dtype=torch.bfloat16),
                k_nope=torch.empty(1, 512, dtype=torch.bfloat16),
                k_pe=torch.empty(1, 64, dtype=torch.bfloat16),
                positions=torch.zeros(1, dtype=torch.int64),
                out_cache_loc=torch.zeros(1, dtype=torch.int32),
            )

        self.assertEqual(
            fused_op.call_args.kwargs["q_out_dtype"], forward_mla_rocm.fp8_dtype
        )


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


if __name__ == "__main__":
    unittest.main()
