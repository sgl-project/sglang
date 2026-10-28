"""CPU tests for the Qwen3.5 Gemma add+RMSNorm+MXFP4 quant producer (ROCm/AITER
gfx950): which norms use it, which batches take it, and how its (fp4x2, E8M0)
pair reaches the projections. The kernel is faked; its numerics are covered by
test/registered/kernels/ops/layernorm/test_gemma_rmsnorm_mxfp4_quant.py."""

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    QuarkW4A4MXFP4,
    _AsmShuffledMxfp4Input,
)
from sglang.srt.models import qwen3_5
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HIDDEN = 512
_KERNEL_MODULE = "sglang.kernels.ops.layernorm.minimax_m3_rmsnorm"
_FP4X2, _E8M0 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu


def _rocm_gfx95(**overrides):
    """qwen3_5 as on ROCm/AITER gfx950 with the Triton Gemma norm."""
    patches = dict(
        _use_aiter=True,
        _is_gfx95=True,
        aiter_dtypes=SimpleNamespace(fp4x2=_FP4X2, fp8_e8m0=_E8M0),
        gemma_rms_norm_uses_rocm_triton=lambda: True,
        get_forward=lambda: SimpleNamespace(sp_active=False),
        is_dp_attention_enabled=lambda: False,
        get_is_capture_mode=lambda: False,
    )
    patches.update(overrides)
    return mock.patch.multiple(qwen3_5, create=True, **patches)


def _mxfp4_linear(use_asm=True):
    linear = nn.Module()
    linear.scheme = object.__new__(QuarkW4A4MXFP4)
    linear.use_aiter_asm_fp4_gemm = use_asm
    return linear


def _pair(m=3, k=_HIDDEN):
    return _AsmShuffledMxfp4Input(
        (
            torch.zeros(m, k // 2, dtype=torch.uint8).view(_FP4X2),
            torch.zeros(256, k // 32, dtype=torch.uint8).view(_E8M0),
        )
    )


class TestNormSelection(CustomTestCase):
    def _norm_for(self, consumer, hidden_size=_HIDDEN):
        config = SimpleNamespace(hidden_size=hidden_size, rms_norm_eps=1e-6)
        return qwen3_5._gemma_rms_norm_for(config, consumer, lambda num_tokens: True)

    def test_mxfp4_consumer_gets_fused_norm(self):
        with _rocm_gfx95():
            norm = self._norm_for(_mxfp4_linear())
        self.assertIsInstance(norm, qwen3_5._GemmaRMSNormMXFP4Quant)
        self.assertEqual(norm.variance_epsilon, 1e-6)

    def test_plain_norm_otherwise(self):
        cases = {
            "no AITER": ({"_use_aiter": False}, _mxfp4_linear(), _HIDDEN),
            "not gfx95": ({"_is_gfx95": False}, _mxfp4_linear(), _HIDDEN),
            "bf16 norm not on the ROCm Triton kernel": (
                {"gemma_rms_norm_uses_rocm_triton": lambda: False},
                _mxfp4_linear(),
                _HIDDEN,
            ),
            "hidden size % 256": ({}, _mxfp4_linear(), 640),
            "hidden size > 8192": ({}, _mxfp4_linear(), 8448),
            "no consumer (MoE layer)": ({}, None, _HIDDEN),
            "non-MXFP4 consumer": ({}, nn.Linear(4, 4), _HIDDEN),
        }
        for name, (overrides, consumer, hidden_size) in cases.items():
            with self.subTest(name), _rocm_gfx95(**overrides):
                norm = self._norm_for(consumer, hidden_size)
            self.assertIs(type(norm), GemmaRMSNorm)

    def test_plain_norm_without_the_kernel(self):
        with _rocm_gfx95(), mock.patch.dict(sys.modules, {_KERNEL_MODULE: None}):
            norm = self._norm_for(_mxfp4_linear())
        self.assertIs(type(norm), GemmaRMSNorm)


class TestNormForward(CustomTestCase):
    def _norm(self, consumes=True):
        norm = qwen3_5._GemmaRMSNormMXFP4Quant(
            _HIDDEN, 1e-6, lambda num_tokens: consumes
        )
        norm.weight.data = norm.weight.data.bfloat16()
        calls = []

        def norm_quant(x, residual, weight, eps):
            calls.append((x, residual, weight, eps))
            m, n = x.shape
            return (
                torch.zeros(m, n // 2, dtype=torch.uint8),
                torch.zeros((m + 255) // 256 * 256, n // 32, dtype=torch.uint8),
                x + residual,
            )

        norm._norm_quant = norm_quant
        return norm, calls

    @staticmethod
    def _inputs(m=3, dtype=torch.bfloat16):
        x = torch.randn(m, _HIDDEN).to(dtype)
        return x, torch.randn(m, _HIDDEN).to(dtype)

    def test_emits_asm_shuffled_pair(self):
        norm, calls = self._norm()
        x, residual = self._inputs()
        with _rocm_gfx95():
            out, residual_out = norm.forward_hip(x, residual)
        self.assertIsInstance(out, _AsmShuffledMxfp4Input)
        x_q, x_scales = out
        self.assertEqual((x_q.dtype, tuple(x_q.shape)), (_FP4X2, (3, _HIDDEN // 2)))
        self.assertEqual(
            (x_scales.dtype, tuple(x_scales.shape)), (_E8M0, (256, _HIDDEN // 32))
        )
        torch.testing.assert_close(residual_out, x + residual, rtol=0, atol=0)
        ((x_in, residual_in, weight, eps),) = calls
        self.assertIs(x_in, x)
        self.assertIs(residual_in, residual)
        self.assertEqual(weight.data_ptr(), norm.weight.data_ptr())
        self.assertEqual(eps, 1e-6)

    def test_falls_back_to_bf16_norm(self):
        x, residual = self._inputs()
        x16, residual16 = self._inputs(dtype=torch.float16)
        cases = {
            "first layer (no residual)": ((x,), True),
            "post-residual addition": ((x, residual, torch.zeros_like(x)), True),
            "consumer declines the batch": ((x, residual), False),
            "fp16 activations": ((x16, residual16), True),
            "non-contiguous input": ((x.t().contiguous().t(), residual), True),
        }
        for name, (args, consumes) in cases.items():
            # Keep the bf16 fallback on torch math: the ROCm Triton norm the
            # real forward_hip runs needs a GPU.
            with (
                self.subTest(name),
                _rocm_gfx95(),
                mock.patch.object(
                    GemmaRMSNorm, "forward_hip", GemmaRMSNorm.forward_native
                ),
            ):
                norm, calls = self._norm(consumes)
                out = norm.forward_hip(*args)
            out = out[0] if isinstance(out, tuple) else out
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(calls, [])


class TestMxfp4Consumers(CustomTestCase):
    def test_asm_input_check(self):
        with _rocm_gfx95():
            self.assertTrue(qwen3_5._mxfp4_asm_input(_mxfp4_linear(), 4))
            self.assertFalse(qwen3_5._mxfp4_asm_input(_mxfp4_linear(), 0))
            self.assertFalse(qwen3_5._mxfp4_asm_input(_mxfp4_linear(False), 4))
        for overrides in (
            {"get_forward": lambda: SimpleNamespace(sp_active=True)},
            {"is_dp_attention_enabled": lambda: True},
        ):
            with self.subTest(overrides), _rocm_gfx95(**overrides):
                self.assertFalse(qwen3_5._mxfp4_asm_input(_mxfp4_linear(), 4))

    def test_pair_reaches_asm_linears_only(self):
        pair = _pair()
        with _rocm_gfx95():
            self.assertIs(
                qwen3_5._select_fused_ar_input_for_linear(pair, _mxfp4_linear()), pair
            )
            with self.assertRaises(TypeError):
                qwen3_5._select_fused_ar_input_for_linear(pair, _mxfp4_linear(False))

    def test_packed_in_proj_takes_the_pair(self):
        gdn = SimpleNamespace(_packed_mxfp4_in_proj=mock.Mock(return_value="out"))
        x_q, x_scales = pair = _pair()
        with _rocm_gfx95():
            out = qwen3_5.Qwen3_5GatedDeltaNet._forward_input_proj(gdn, pair)
        self.assertEqual(out, "out")
        gdn._packed_mxfp4_in_proj.assert_called_once_with(x_q, x_scales)

    def test_packed_in_proj_applies(self):
        def applies(num_tokens, capture=False, **gdn_overrides):
            gdn = dict(
                _fused_in_proj_weight=torch.empty(1),
                _fused_in_proj_mxfp4=True,
                _fused_fp8_in_proj_sources_valid=lambda: True,
            )
            gdn.update(gdn_overrides)
            with _rocm_gfx95(get_is_capture_mode=lambda: capture):
                return qwen3_5.Qwen3_5GatedDeltaNet._packed_mxfp4_in_proj_applies(
                    SimpleNamespace(**gdn), num_tokens
                )

        # The row caps of the packed GEMM: 64 eager, 256 under graph capture.
        self.assertTrue(applies(1) and applies(64))
        self.assertFalse(applies(0) or applies(65))
        self.assertTrue(applies(256, capture=True))
        self.assertFalse(applies(257, capture=True))
        self.assertFalse(applies(1, _fused_in_proj_weight=None))
        self.assertFalse(applies(1, _fused_in_proj_mxfp4=False))
        self.assertFalse(applies(1, _fused_fp8_in_proj_sources_valid=lambda: False))


if __name__ == "__main__":
    unittest.main()
