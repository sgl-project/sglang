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
from unittest import mock

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


_GLUON_OK = forward_mla_rocm._gluon_mla_decode_gfx950_ok
_GLUON_LAYER = dict(
    kv_cache_dtype="fp8_e4m3",
    kv_lora_rank=512,
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    v_head_dim=256,
    num_local_heads=8,
    dcp_size=1,
)


class TestGluonMlaDecodeGate(CustomTestCase):
    """Which layers may swap the aiter decode prologue for the gfx950 Gluon kernel."""

    def setUp(self):
        self.calls = []
        self.kernels = SimpleNamespace(
            gluon_mla_absorb_rope_cache_supported=lambda *args: True,
            gluon_mla_absorb_rope_cache=lambda *args: (
                self.calls.append("gluon") or "q_cat"
            ),
        )
        exec_cfg = SimpleNamespace(kernel=SimpleNamespace(dsa_decode_backend="triton"))
        for name, value in (
            ("_gluon_mla_absorb", self.kernels),
            ("get_exec", lambda: exec_cfg),
            (
                "get_kv_layers",
                lambda: SimpleNamespace(read=lambda i: SimpleNamespace(key=lambda: i)),
            ),
            ("fp8_dtype", torch.float8_e4m3fn),
            (
                "fused_fp8_bmm_rope_cat_and_cache_mla",
                lambda *args, **kwargs: self.calls.append("aiter") or ("q",) * 4,
            ),
        ):
            p = mock.patch.object(forward_mla_rocm, name, value, create=True)
            p.start()
            self.addCleanup(p.stop)
        self.exec_cfg = exec_cfg

    def test_takes_gluon_when_every_term_holds(self):
        for heads in (8, 16):
            with self.subTest(heads=heads):
                self.assertTrue(_GLUON_OK(**{**_GLUON_LAYER, "num_local_heads": heads}))

    def test_each_term_alone_turns_it_off(self):
        for name, broken in (
            ("kv_cache_dtype", "auto"),
            ("kv_lora_rank", 448),
            ("qk_nope_head_dim", 128),
            ("qk_rope_head_dim", 0),
            ("v_head_dim", 128),
            ("num_local_heads", 12),
            ("dcp_size", 2),
        ):
            with self.subTest(term=name):
                self.assertFalse(_GLUON_OK(**{**_GLUON_LAYER, name: broken}))
        # aiter and tilelang decode read the roped k_pe, which this kernel skips.
        for backend in ("aiter", "tilelang"):
            self.exec_cfg.kernel.dsa_decode_backend = backend
            self.assertFalse(_GLUON_OK(**_GLUON_LAYER))
        self.exec_cfg.kernel.dsa_decode_backend = "triton"
        with mock.patch.object(forward_mla_rocm, "_gluon_mla_absorb", None):
            self.assertFalse(_GLUON_OK(**_GLUON_LAYER))

    def test_outside_the_gate_the_aiter_kernel_runs(self):
        attn = SimpleNamespace(
            use_gluon_mla_decode_gfx950=True,
            rotary_emb=SimpleNamespace(
                is_neox_style=False, cos_cache=None, sin_cache=None
            ),
            attn_mqa=SimpleNamespace(layer_id=0, k_scale=None),
            w_kc=torch.empty(8, 192, 512),
            w_scale=None,
            kv_cache_dtype="fp8_e4m3",
        )
        args = (attn,) + (torch.empty(1, 1, 1),) * 6
        self.assertEqual(
            forward_mla_rocm._fused_bmm_rope_cat_and_cache(
                *args, decode_or_verify=True
            )[0],
            "q_cat",
        )
        # Draft extend also takes the fused path upstream; the Gluon kernel is
        # only enabled for decode and target verify.
        self.calls.clear()
        forward_mla_rocm._fused_bmm_rope_cat_and_cache(*args, decode_or_verify=False)
        self.assertEqual(self.calls, ["aiter"])
        for flip in (
            lambda: setattr(attn, "use_gluon_mla_decode_gfx950", False),
            lambda: setattr(attn.rotary_emb, "is_neox_style", True),
            lambda: setattr(
                self.kernels,
                "gluon_mla_absorb_rope_cache_supported",
                lambda *args: False,
            ),
        ):
            flip()
            self.calls.clear()
            forward_mla_rocm._fused_bmm_rope_cat_and_cache(*args, decode_or_verify=True)
            self.assertEqual(self.calls, ["aiter"])


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
