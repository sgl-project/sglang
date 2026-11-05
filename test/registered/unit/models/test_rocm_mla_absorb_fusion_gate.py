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

from sglang.srt.models import kimi_k3_rocm_mla
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.srt.models.kimi_k3_rocm_mla import KimiK3RocmMlaQCacheWriter
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


class TestKimiK3MlaQCacheWriterGate(CustomTestCase):
    """The K3 writer replaces the split chain only where attention never reads its k,
    and the cache holds the bytes the split chain would have written."""

    def setUp(self):
        self.writer = KimiK3RocmMlaQCacheWriter(kv_lora_rank=512, qk_rope_head_dim=64)
        self.fused_q = torch.zeros(2, 12, 576, dtype=torch.bfloat16)
        # The kernels are the boundary; the gate in front of them is under test.
        self.writer._write_aiter = lambda **_: self.fused_q

    def _write(self, *, attention_backend, is_target_verify, k_scale_float=1.0):
        return self.writer.write(
            q_nope=torch.zeros(2, 12, 512, dtype=torch.bfloat16),
            q_pe=torch.zeros(2, 12, 64, dtype=torch.bfloat16),
            k_nope=torch.zeros(2, 1, 512, dtype=torch.bfloat16),
            k_pe=torch.zeros(2, 1, 64, dtype=torch.bfloat16),
            out_cache_loc=torch.arange(2, dtype=torch.int64),
            kv_cache=torch.zeros(4, 1, 576, dtype=torch.bfloat16),
            k_scale_float=k_scale_float,
            attention_backend=attention_backend,
            is_target_verify=is_target_verify,
        )

    def test_triton_target_verify_keeps_the_split_chain(self):
        """Triton target-verify runs the extend kernel, which reads k;
        fusing it would feed that kernel the uninitialized k placeholder."""
        self.assertIsNone(
            self._write(attention_backend="triton", is_target_verify=True)
        )

    def test_fused_routes(self):
        for backend, verify in (("aiter", False), ("aiter", True), ("triton", False)):
            with self.subTest(backend=backend, verify=verify):
                fused = self._write(attention_backend=backend, is_target_verify=verify)
                self.assertIs(fused[0], self.fused_q)
                self.assertEqual(tuple(fused[1].shape), (2, 1, 576))

    def test_non_unit_kv_scale_keeps_the_split_chain(self):
        """The split chain's AITER cache write casts without the KV scale,
        while the fused kernels divide by it; their bytes differ unless it is 1."""
        for k_scale_float in (None, 1.0):
            with self.subTest(k_scale_float=k_scale_float):
                self.assertIsNotNone(
                    self._write(
                        attention_backend="aiter",
                        is_target_verify=False,
                        k_scale_float=k_scale_float,
                    )
                )
        self.assertIsNone(
            self._write(
                attention_backend="aiter", is_target_verify=False, k_scale_float=2.0
            )
        )

    def test_missing_triton_fallback_keeps_the_split_chain(self):
        """Off gfx950, an AITER build without the no-RoPE Triton kernel must fall
        back to the split chain instead of failing the decode step."""
        self.writer._write_aiter = lambda **_: None
        with mock.patch.object(kimi_k3_rocm_mla, "_triton_q_cache_op", lambda: None):
            self.assertIsNone(
                self._write(attention_backend="aiter", is_target_verify=False)
            )

    def test_other_backends_keep_the_split_chain(self):
        for verify in (False, True):
            with self.subTest(verify=verify):
                self.assertIsNone(
                    self._write(attention_backend="dsa", is_target_verify=verify)
                )


class TestMlaQCacheWriterCallSite(CustomTestCase):
    """forward_absorb_rocm_core consults the writer only for non-DCP decode
    and target-verify; every other step keeps the split chain."""

    def _call(self, *, mode, dcp_enabled=False, owner=True):
        writer = mock.Mock()
        kv_layer = SimpleNamespace(owner=owner, key=lambda: None)
        attn = SimpleNamespace(
            rocm_mla_q_cache_writer=writer,
            attn_mqa=SimpleNamespace(layer_id=0, k_scale_float=1.0),
            current_attention_backend="aiter",
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: mode == "decode",
                is_target_verify=lambda: mode == "target_verify",
            ),
            out_cache_loc=torch.arange(2),
        )
        with (
            mock.patch.object(
                forward_mla_rocm,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=dcp_enabled),
            ),
            mock.patch.object(
                forward_mla_rocm,
                "get_kv_layers",
                return_value=SimpleNamespace(read=lambda _: kv_layer),
            ),
        ):
            result = forward_mla_rocm.DeepseekMLARocmForwardMixin._maybe_fused_q_cache(
                attn,
                q_nope_out=None,
                q_pe=None,
                k_nope=None,
                k_pe=None,
                forward_batch=forward_batch,
            )
        return result, writer.write

    def test_decode_and_target_verify_use_the_writer(self):
        for mode in ("decode", "target_verify"):
            with self.subTest(mode=mode):
                result, write = self._call(mode=mode)
                write.assert_called_once()
                self.assertIs(result, write.return_value)
                self.assertEqual(
                    write.call_args.kwargs["is_target_verify"], mode == "target_verify"
                )

    def test_dcp_and_extend_keep_the_split_chain(self):
        """DCP decode all-gathers q from the split chain, and extend is not covered."""
        for mode, dcp_enabled in (("decode", True), ("extend", False)):
            with self.subTest(mode=mode, dcp_enabled=dcp_enabled):
                result, write = self._call(mode=mode, dcp_enabled=dcp_enabled)
                self.assertIsNone(result)
                write.assert_not_called()

    def test_non_owner_sub_pool_keeps_the_split_chain(self):
        """The writer scatters at out_cache_loc, but a non-owner sub-pool's layer
        is written at the batch's swa_loc; fusing it would write the wrong rows."""
        result, write = self._call(mode="decode", owner=False)
        self.assertIsNone(result)
        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
