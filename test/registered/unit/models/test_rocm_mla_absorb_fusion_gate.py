"""The gate that picks the fused q-absorb + RoPE + KV-write kernel.

Every term names a branch that one of the two kernels it replaces would
otherwise have taken, so the fused path is only equivalent where all of them
hold. A term quietly dropped here would route a shape the kernel does not
implement, and the wrong answer would look like a plausible one.

The caller carries one further term, ``not q_replicate_active``, which needs a
ForwardBatch to evaluate and so is not part of this predicate.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_CAN_FUSE = forward_mla_rocm._can_fuse_bmm_rope_cat_and_cache
_ABSORB_CONFIG = forward_mla_rocm._absorb_bmm_config
_VALUE_QUANT_ENV = envs.SGLANG_OPT_HIP_FUSE_MLA_VALUE_MXFP4_QUANT
_CAN_FUSE_VALUE_QUANT = forward_mla_rocm._can_fuse_rocm_mla_value_mxfp4_quant


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


def _value_attn(**overrides):
    """An attention module the value-quant gate accepts, before the caller breaks one."""
    attn = SimpleNamespace(
        current_attention_backend="aiter",
        use_dsa=False,
        w_vc=torch.empty(1, dtype=torch.uint8),
        w_scale_v=torch.empty(1, dtype=torch.uint8),
        o_proj=SimpleNamespace(weight=torch.empty(1, dtype=torch.uint8)),
        kv_lora_rank=512,
        v_head_dim=128,
    )
    for k, v in overrides.items():
        setattr(attn, k, v)
    return attn


def _forward_batch(is_decode_or_idle=True):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: is_decode_or_idle)
    )


class TestValueQuantFusionGate(CustomTestCase):
    """The gate that picks the fused value BMM + MXFP4 output-quant kernel.

    The kernel emits the MXFP4 o_proj input directly, skipping the BF16 value output;
    a kv_b_proj LoRA delta or a non-MXFP4 o_proj needs that output and stays split.
    """

    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            mock.patch.object(forward_mla_rocm, "_use_aiter_gfx95", True)
        )
        stack.enter_context(
            mock.patch.object(
                forward_mla_rocm, "is_kv_b_lora_active", return_value=False
            )
        )
        stack.enter_context(_VALUE_QUANT_ENV.override(True))

    def test_takes_the_fused_path_when_every_term_holds(self):
        self.assertTrue(_CAN_FUSE_VALUE_QUANT(_value_attn(), _forward_batch()))

    def test_active_kv_b_lora_turns_it_off(self):
        """The fused kernel has no hook for the kv_b_proj LoRA delta on the value output."""
        with mock.patch.object(
            forward_mla_rocm, "is_kv_b_lora_active", return_value=True
        ):
            self.assertFalse(_CAN_FUSE_VALUE_QUANT(_value_attn(), _forward_batch()))

    def test_each_term_alone_turns_it_off(self):
        # The kernel is decode-only on the AITER MLA backend, reads scaled MXFP4 W_vc,
        # feeds an MXFP4 o_proj, and is pinned to the Kimi N=128, K=512 shape.
        for name, broken in (
            ("current_attention_backend", "triton"),
            ("use_dsa", True),
            ("w_vc", torch.empty(1, dtype=torch.bfloat16)),
            ("w_scale_v", None),
            ("o_proj", SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))),
            ("kv_lora_rank", 576),
            ("v_head_dim", 64),
        ):
            with self.subTest(term=name):
                self.assertFalse(
                    _CAN_FUSE_VALUE_QUANT(
                        _value_attn(**{name: broken}), _forward_batch()
                    )
                )
        with self.subTest(term="forward_mode"):
            self.assertFalse(
                _CAN_FUSE_VALUE_QUANT(_value_attn(), _forward_batch(False))
            )
        with self.subTest(term="env"):
            with _VALUE_QUANT_ENV.override(False):
                self.assertFalse(_CAN_FUSE_VALUE_QUANT(_value_attn(), _forward_batch()))


if __name__ == "__main__":
    unittest.main()
