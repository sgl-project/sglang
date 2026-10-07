"""Unit tests for FlyDSL FP8 MLA prefill operand routing."""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend, fp8_dtype
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    DeepseekMHAForwardMixin,
)
from sglang.srt.models.deepseek_common.utils import _is_hip
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAiterFlyDSLPrefill(unittest.TestCase):
    def setUp(self):
        self.backend = AiterAttnBackend.__new__(AiterAttnBackend)
        self.backend.use_mla_flydsl_fp8_prefill = True
        self.q = torch.empty((8, 12, 192), dtype=torch.bfloat16)

    def test_accepts_bf16_operands(self):
        k = torch.empty((16, 12, 192), dtype=torch.bfloat16)
        v = torch.empty((16, 12, 128), dtype=torch.bfloat16)
        self.assertTrue(self.backend._mla_flydsl_fp8_prefill_applicable(self.q, k, v))

    def test_accepts_direct_fp8_kv(self):
        k = torch.empty((16, 12, 192), dtype=fp8_dtype)
        v = torch.empty((16, 12, 128), dtype=fp8_dtype)
        self.assertTrue(self.backend._mla_flydsl_fp8_prefill_applicable(self.q, k, v))

    def test_rejects_mixed_kv_dtypes(self):
        k = torch.empty((16, 12, 192), dtype=fp8_dtype)
        v = torch.empty((16, 12, 128), dtype=torch.bfloat16)
        self.assertFalse(self.backend._mla_flydsl_fp8_prefill_applicable(self.q, k, v))


class _PrefixProj:
    def __init__(self, dtype):
        self.weight = torch.empty(1, dtype=dtype)
        self.calls = []

    def __call__(self, args):
        self.calls.append(args)
        k = torch.empty((args[1].shape[0], args[1].shape[1], args[2] + 64))
        v = torch.empty((args[1].shape[0], args[1].shape[1], args[3]))
        return (k, v), None


class TestPrefixKvFp8Projection(unittest.TestCase):
    def _module(self, dtype):
        module = DeepseekMHAForwardMixin()
        module.kv_b_proj = _PrefixProj(dtype)
        module.num_local_heads = 12
        module.qk_nope_head_dim = 128
        module.v_head_dim = 128
        return module

    def test_stays_off_without_the_flag_or_mxfp4_weights(self):
        module = self._module(torch.float32)
        kv_a = torch.empty((4, 512))
        k_pe = torch.empty((4, 1, 64))
        with envs.SGLANG_AITER_MLA_FLYDSL_FUSED_KV_PROJ.override(False):
            self.assertIsNone(module._project_prefix_kv_fp8(kv_a, k_pe))
        with envs.SGLANG_AITER_MLA_FLYDSL_FUSED_KV_PROJ.override(True):
            self.assertIsNone(module._project_prefix_kv_fp8(kv_a, k_pe))
        self.assertEqual(module.kv_b_proj.calls, [])

    def test_dispatches_mxfp4_weights_when_enabled(self):
        module = self._module(torch.uint8)
        kv_a = torch.empty((4, 1, 512))
        k_pe = torch.empty((4, 1, 64))
        with envs.SGLANG_AITER_MLA_FLYDSL_FUSED_KV_PROJ.override(True):
            projected = module._project_prefix_kv_fp8(kv_a, k_pe)
        if not _is_hip:
            self.assertIsNone(projected)
            return
        self.assertIsNotNone(projected)
        k, v = projected
        self.assertEqual(k.shape, (4, 12, 192))
        self.assertEqual(v.shape, (4, 12, 128))
        self.assertEqual(len(module.kv_b_proj.calls), 1)


if __name__ == "__main__":
    unittest.main()
