"""The gate that picks the fused q-absorb + RoPE + KV-write kernel.

Every term names a branch that one of the two kernels it replaces would
otherwise have taken, so the fused path is only equivalent where all of them
hold. A term quietly dropped here would route a shape the kernel does not
implement, and the wrong answer would look like a plausible one.

The caller carries one further term, ``not q_replicate_active``, which needs a
ForwardBatch to evaluate and so is not part of this predicate.
"""

import unittest
from functools import partial
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
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
        kv_lora_rank=512,
        kv_cache_dtype="fp8_e4m3",
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
        saved_gfx950 = forward_mla_rocm._use_aiter_gfx950
        forward_mla_rocm._use_aiter_gfx950 = True
        self.addCleanup(setattr, forward_mla_rocm, "_use_aiter_gfx950", saved_gfx950)
        self._parallel = _patch_dcp(self, dcp_enabled=False)
        self._exec = SimpleNamespace(
            kernel=SimpleNamespace(dsa_decode_backend="tilelang"),
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend="full")
                )
            ),
        )
        saved_exec = forward_mla_rocm.get_exec
        forward_mla_rocm.get_exec = lambda: self._exec
        self.addCleanup(setattr, forward_mla_rocm, "get_exec", saved_exec)
        self._route = partial(
            forward_mla_rocm._fuse_bmm_rope_cache, is_capture_mode=True
        )
        self._memory = SimpleNamespace(enable_hisparse=False)
        saved_memory = forward_mla_rocm.get_memory
        forward_mla_rocm.get_memory = lambda: self._memory
        self.addCleanup(setattr, forward_mla_rocm, "get_memory", saved_memory)

    def test_verify_routing_boundaries_and_concurrency_controls(self):
        for m in (4, 8, 16, 32, 33, 64, 65, 128, 129, 256, 512, 1024):
            with self.subTest(m=m):
                self.assertEqual(
                    self._route(
                        _attn(),
                        torch.empty(m, 8, 192, dtype=torch.bfloat16),
                        ForwardMode.TARGET_VERIFY,
                        False,
                    ),
                    m != 64,
                )

    def test_eager_and_other_graph_backends_keep_fusion(self):
        q = torch.empty(64, 8, 192, dtype=torch.bfloat16)
        self.assertTrue(
            forward_mla_rocm._fuse_bmm_rope_cache(
                _attn(), q, ForwardMode.TARGET_VERIFY, False, is_capture_mode=False
            )
        )
        for backend in ("breakable", "tc_piecewise", "disabled"):
            self._exec.graph.cuda_graph_config.decode.backend = backend
            self.assertTrue(self._route(_attn(), q, ForwardMode.TARGET_VERIFY, False))
        self._exec.graph.cuda_graph_config = None
        self.assertTrue(self._route(_attn(), q, ForwardMode.TARGET_VERIFY, False))

    def test_other_forward_modes_keep_fusion(self):
        q = torch.empty(64, 8, 192, dtype=torch.bfloat16)
        for mode in ForwardMode:
            with self.subTest(mode=mode):
                self.assertEqual(
                    self._route(_attn(), q, mode, False),
                    mode != ForwardMode.TARGET_VERIFY,
                )

    def test_unqualified_shapes_and_precision_keep_fusion(self):
        for heads, k, dtype, overrides in (
            (16, 192, torch.bfloat16, {}),
            (8, 128, torch.bfloat16, {}),
            (8, 192, torch.float16, {}),
            (8, 192, torch.bfloat16, {"kv_lora_rank": 256}),
            (8, 192, torch.bfloat16, {"kv_cache_dtype": "bfloat16"}),
        ):
            with self.subTest(heads=heads, k=k, dtype=dtype, overrides=overrides):
                self.assertTrue(
                    self._route(
                        _attn(**overrides),
                        torch.empty(64, heads, k, dtype=dtype),
                        ForwardMode.TARGET_VERIFY,
                        False,
                    )
                )
        self._exec.kernel.dsa_decode_backend = "triton"
        self.assertTrue(
            self._route(
                _attn(),
                torch.empty(64, 8, 192, dtype=torch.bfloat16),
                ForwardMode.TARGET_VERIFY,
                False,
            )
        )

    def test_unqualified_device_and_hisparse_keep_fusion(self):
        q = torch.empty(64, 8, 192, dtype=torch.bfloat16)
        route = self._route
        forward_mla_rocm._use_aiter_gfx950 = False
        self.assertTrue(route(_attn(), q, ForwardMode.TARGET_VERIFY, False))
        forward_mla_rocm._use_aiter_gfx950 = True
        self._memory.enable_hisparse = True
        self.assertTrue(route(_attn(), q, ForwardMode.TARGET_VERIFY, False))

    def test_existing_fallbacks_still_win_over_the_row_policy(self):
        q = torch.empty(32, 8, 192, dtype=torch.bfloat16)
        route = self._route
        self.assertFalse(route(_attn(), q, ForwardMode.TARGET_VERIFY, True))
        self._parallel.dcp_enabled = True
        self.assertFalse(route(_attn(), q, ForwardMode.TARGET_VERIFY, False))
        self._parallel.dcp_enabled = False
        for overrides in (
            {"kv_b_proj": SimpleNamespace(set_lora=True)},
            {"use_deep_gemm_bmm": True},
            {"current_attention_backend": "triton"},
            {"w_kc": SimpleNamespace(dtype=torch.bfloat16)},
        ):
            with self.subTest(overrides=overrides):
                self.assertFalse(
                    route(_attn(**overrides), q, ForwardMode.TARGET_VERIFY, False)
                )
        saved = forward_mla_rocm._SGLANG_EXPERIMENTAL_LORA_OPTI
        try:
            forward_mla_rocm._SGLANG_EXPERIMENTAL_LORA_OPTI = True
            self.assertFalse(route(_attn(), q, ForwardMode.TARGET_VERIFY, False))
        finally:
            forward_mla_rocm._SGLANG_EXPERIMENTAL_LORA_OPTI = saved

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
