"""CPU tests for AITER FlyDSL FP8 MLA prefill routing and metadata."""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention import aiter_backend as mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _backend(*, use_flydsl: bool, use_asm: bool = False):
    backend = mod.AiterAttnBackend.__new__(mod.AiterAttnBackend)
    backend.use_flydsl_fp8_prefill_attn = use_flydsl
    backend.use_asm_fp8_prefill_attn = use_asm
    return backend


def _metadata():
    indptr = torch.tensor([0, 2], dtype=torch.int32)
    return mod.MlaPrefillPsMetadata(
        qo_indptr=indptr,
        kv_indptr=indptr,
        max_q_len=2,
        max_kv_len=2,
        is_causal=True,
        need_lse=False,
        qo_indptr_cpu=indptr,
        kv_indptr_cpu=indptr,
        kv_lens_cpu=torch.tensor([2], dtype=torch.int32),
        num_kv_tokens=2,
        exact_partial_count=True,
    )


class TestFlyDslFp8QuantRows(unittest.TestCase):
    def test_uses_largest_divisor_up_to_target(self):
        for num_vectors, expected in (
            (1, 1),
            (16, 16),
            (256, 256),
            (300, 150),
            (512, 256),
            (257, 1),
        ):
            with self.subTest(num_vectors=num_vectors):
                self.assertEqual(mod._flydsl_fp8_quant_rows(num_vectors), expected)


class TestFlyDslFp8FeatureFlag(unittest.TestCase):
    def test_requires_master_fp8_and_flydsl_flags(self):
        for fp8_enabled, flydsl_enabled, expected in (
            (False, False, False),
            (False, True, False),
            (True, False, False),
            (True, True, True),
        ):
            with (
                self.subTest(
                    fp8_enabled=fp8_enabled,
                    flydsl_enabled=flydsl_enabled,
                ),
                mock.patch.object(mod, "_use_fp8_prefill_attn", fp8_enabled),
                mock.patch.object(mod, "get_bool_env_var", return_value=flydsl_enabled),
            ):
                self.assertIs(mod._flydsl_fp8_prefill_requested(), expected)


class TestFlyDslFp8PrefillDispatch(unittest.TestCase):
    def setUp(self):
        self.q = torch.zeros((2, 8, 576), dtype=torch.bfloat16)
        self.k = torch.zeros_like(self.q)
        self.v = torch.zeros((2, 8, 512), dtype=torch.bfloat16)
        self.layer = mock.Mock()
        self.ps = _metadata()

    def test_supported_shape_dispatches_to_flydsl(self):
        backend = _backend(use_flydsl=True)
        expected = (mock.sentinel.output, mock.sentinel.lse)

        with (
            mock.patch.object(
                mod, "flydsl_flash_attn_fp8_supported", return_value=True, create=True
            ) as is_supported,
            mock.patch.object(
                backend,
                "_mla_flydsl_fp8_prefill_attn",
                return_value=expected,
            ) as flydsl,
            mock.patch.object(backend, "_mla_varlen_prefill_attn") as varlen_fallback,
        ):
            result = backend._mla_fp8_prefill_attn_ps(
                self.q, self.k, self.v, self.layer, self.ps
            )

        self.assertEqual(result, expected)
        is_supported.assert_called_once_with(
            self.q.device,
            self.q.shape[-2],
            self.k.shape[-2],
            self.q.shape[-1],
            self.v.shape[-1],
            dtype=mod.fp8_dtype,
        )
        flydsl.assert_called_once_with(self.q, self.k, self.v, self.layer, self.ps)
        varlen_fallback.assert_not_called()

    def test_unsupported_shape_uses_existing_bf16_fallback(self):
        backend = _backend(use_flydsl=True, use_asm=False)
        expected = (mock.sentinel.output, mock.sentinel.lse)

        with (
            mock.patch.object(
                mod, "flydsl_flash_attn_fp8_supported", return_value=False, create=True
            ),
            mock.patch.object(backend, "_mla_flydsl_fp8_prefill_attn") as flydsl,
            mock.patch.object(
                backend,
                "_mla_varlen_prefill_attn",
                return_value=expected,
            ) as varlen_fallback,
        ):
            result = backend._mla_fp8_prefill_attn_ps(
                self.q, self.k, self.v, self.layer, self.ps
            )

        self.assertEqual(result, expected)
        flydsl.assert_not_called()
        varlen_fallback.assert_called_once_with(
            self.q, self.k, self.v, self.layer, self.ps
        )


class TestFlyDslFp8PrefillMetadata(unittest.TestCase):
    def _build(self, backend):
        return backend._build_prefill_ps_metadata(
            qo_indptr=torch.tensor([0, 2], dtype=torch.int32),
            kv_indptr=torch.tensor([0, 4], dtype=torch.int32),
            kv_lens_cpu=torch.tensor([4], dtype=torch.int32),
            num_kv_tokens=4,
            max_q_len=2,
            is_causal=True,
            need_lse=False,
        )

    def test_flydsl_defers_asm_metadata(self):
        backend = _backend(use_flydsl=True)

        with mock.patch.object(
            backend, "_materialize_asm_prefill_ps_metadata"
        ) as materialize:
            ps = self._build(backend)

        materialize.assert_not_called()
        self.assertIsNone(ps.work_metadata)

    def test_non_flydsl_materializes_asm_metadata(self):
        backend = _backend(use_flydsl=False)

        with mock.patch.object(
            backend, "_materialize_asm_prefill_ps_metadata"
        ) as materialize:
            ps = self._build(backend)

        materialize.assert_called_once_with(ps)


if __name__ == "__main__":
    unittest.main()
