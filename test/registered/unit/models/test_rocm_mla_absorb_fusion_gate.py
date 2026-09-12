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

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CAN_FUSE = forward_mla_rocm._can_fuse_bmm_rope_cat_and_cache


def _attn(**overrides):
    """An attention module every term accepts, before the caller breaks one."""
    attn = SimpleNamespace(
        w_kc=SimpleNamespace(dtype=torch.float8_e4m3fn),
        rotary_emb=object(),
        current_attention_backend="dsa",
        use_deep_gemm_bmm=False,
        kv_b_proj=SimpleNamespace(set_lora=False),
        _skip_rope_for_dsa_tilelang_fused=lambda: True,
    )
    for k, v in overrides.items():
        setattr(attn, k, v)
    return attn


def _patch_dcp(case, *, dcp_enabled: bool):
    """Give the gate a parallel context; the real one needs an initialized runtime."""
    parallel = SimpleNamespace(dcp_enabled=dcp_enabled)
    saved = forward_mla_rocm.get_parallel
    forward_mla_rocm.get_parallel = lambda: parallel
    case.addCleanup(setattr, forward_mla_rocm, "get_parallel", saved)
    return parallel


class TestFusedAbsorbGate(CustomTestCase):
    def setUp(self):
        # The first term is the platform, and it short-circuits everything else;
        # patch it so the remaining terms are reachable off gfx95.
        self._saved = forward_mla_rocm._use_aiter_gfx95
        forward_mla_rocm._use_aiter_gfx95 = True
        self.addCleanup(setattr, forward_mla_rocm, "_use_aiter_gfx95", self._saved)
        self._parallel = _patch_dcp(self, dcp_enabled=False)

    def test_dcp_turns_it_off(self):
        """DCP decode all-gathers q_nope_out, which this path never produces.

        Without this term the fused branch sets q_nope_out to None and
        all_gather_q_for_mla_decode dereferences it.
        """
        self._parallel.dcp_enabled = True
        self.assertFalse(_CAN_FUSE(_attn()))

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


if __name__ == "__main__":
    unittest.main()
