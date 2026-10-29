"""Unit tests for srt/models/kimi_k3_rocm_{quant,moe} and the K3 weight-merge guard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.models.kimi_k3 import _is_unquantized_mergeable
from sglang.srt.models.kimi_k3_rocm_moe import build_tuned_front_gemm
from sglang.srt.models.kimi_k3_rocm_quant import _k3_channel_fp8_to_bf16
from sglang.test.test_utils import CustomTestCase


def _per_channel_fp8(out_features: int, in_features: int):
    torch.manual_seed(0)
    dense = torch.randn(out_features, in_features, dtype=torch.float32)
    # Spread the per-channel magnitudes so a wrongly-broadcast scale cannot
    # coincidentally reproduce the right answer.
    dense *= torch.logspace(-1, 1, out_features).unsqueeze(1)
    scale = dense.abs().amax(dim=1, keepdim=True) / 448.0
    weight = (dense / scale).to(torch.float8_e4m3fn)
    return weight, scale, dense


class TestChannelFp8ToBf16(CustomTestCase):
    """aiter's batched absorb GEMM takes ``w_scale`` as a scalar, so a
    per-channel vector cannot reach it. The scale axis is the GEMM's
    contraction axis, so requantizing to one per-tensor scale is lossy --
    ~2.3% relative error on real K3 weights. Dequantizing to bf16 costs ~0.12%
    and lets the absorb use its bf16 path with w_scale at the 1.0 default."""

    def test_dequantizes_every_channel_with_its_own_scale(self):
        weight, scale, dense = _per_channel_fp8(out_features=8, in_features=16)
        module = SimpleNamespace(weight_scale=scale.squeeze(1))

        out = _k3_channel_fp8_to_bf16(module, weight)

        self.assertEqual(out.dtype, torch.bfloat16)
        # Only the original fp8 rounding plus the bf16 cast separate this from
        # the checkpoint's own values; no second quantization.
        exact = weight.to(torch.float32) * scale
        torch.testing.assert_close(out.float(), exact, rtol=0.01, atol=0.0)
        # Guard the failure this replaced: one shared scale would leave the
        # low-magnitude channels far from their true values.
        worst = ((out.float() - dense).abs().sum(1) / dense.abs().sum(1)).max()
        self.assertLess(worst.item(), 0.05)

    def test_accepts_a_column_vector_scale(self):
        """Quark serializes the scale as [out]; callers may hold [out, 1]."""
        weight, scale, _ = _per_channel_fp8(out_features=8, in_features=16)

        flat = _k3_channel_fp8_to_bf16(
            SimpleNamespace(weight_scale=scale.squeeze(1)), weight
        )
        column = _k3_channel_fp8_to_bf16(
            SimpleNamespace(weight_scale=scale.clone()), weight
        )

        torch.testing.assert_close(flat, column)


class TestMergeDtypeGuard(CustomTestCase):
    """``_merge_weights_as_views`` cats ``.weight`` alone, so fusing a quantized
    projection silently drops the scale tensors that give its integer payload
    meaning. Only dtypes that carry their full value in ``.weight`` may fuse."""

    def test_rejects_weights_that_carry_a_separate_scale(self):
        for dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8):
            with self.subTest(dtype=dtype):
                weights = [torch.zeros(4, 4).to(dtype) for _ in range(2)]
                self.assertFalse(_is_unquantized_mergeable(weights))

    def test_accepts_plain_float_weights(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                weights = [torch.zeros(4, 4, dtype=dtype) for _ in range(2)]
                self.assertTrue(_is_unquantized_mergeable(weights))

    def test_rejects_mixed_dtypes(self):
        weights = [
            torch.zeros(4, 4, dtype=torch.bfloat16),
            torch.zeros(4, 4, dtype=torch.float16),
        ]
        self.assertFalse(_is_unquantized_mergeable(weights))

    def test_rejects_float32(self):
        """The merged buffer is only built for the two half dtypes the fused
        KDA/MoE kernels read; fp32 must not slip in through a dtype-agreement
        check that only asks whether the operands match each other."""
        weights = [torch.zeros(4, 4, dtype=torch.float32) for _ in range(2)]
        self.assertFalse(_is_unquantized_mergeable(weights))


def _shaped(rows: int, cols: int, dtype=torch.bfloat16) -> torch.Tensor:
    # Only shape and dtype are read; expand keeps the 6016x7168 front at one element.
    return torch.zeros(1, dtype=dtype).expand(rows, cols)


def _fake_aiter(*, gfx: str = "gfx950", has_rows: bool = True) -> dict:
    tuned_gemm = types.ModuleType("aiter.tuned_gemm")
    tuned_gemm.tgemm = SimpleNamespace(mm=lambda **kwargs: None)
    tuned_gemm.get_GEMM_A16W16_tuned_config = lambda **kwargs: (
        {"libtype": "flydsl"} if has_rows else None
    )
    chip_info = types.ModuleType("aiter.jit.utils.chip_info")
    chip_info.get_gfx = lambda: gfx
    return {
        "aiter": types.ModuleType("aiter"),
        "aiter.tuned_gemm": tuned_gemm,
        "aiter.jit": types.ModuleType("aiter.jit"),
        "aiter.jit.utils": types.ModuleType("aiter.jit.utils"),
        "aiter.jit.utils.chip_info": chip_info,
    }


class TestTunedFrontGemm(CustomTestCase):
    """AITER has tuned rows for one bf16 front shape (TP8) on gfx950 only;
    the builder declines every other target, and a broken AITER fails the load."""

    def _build(self, *, use_aiter=True, enabled=True, weight=None, aiter=None):
        with (
            patch.dict(sys.modules, _fake_aiter() if aiter is None else aiter),
            envs.SGLANG_USE_AITER.override(use_aiter),
            envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT.override(enabled),
            envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MIN_TOKENS.override(48),
            envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MAX_TOKENS.override(192),
        ):
            return build_tuned_front_gemm(
                weight=_shaped(6016, 7168) if weight is None else weight
            )

    def test_covers_the_inclusive_env_window(self):
        tuned = self._build()
        for num_tokens, covered in ((47, False), (48, True), (192, True), (193, False)):
            with self.subTest(num_tokens=num_tokens):
                self.assertIs(tuned.covers(num_tokens), covered)

    def test_declines_outside_the_tuned_target(self):
        no_aiter = {"aiter": None, "aiter.tuned_gemm": None}
        for name, kwargs in (
            ("SGLANG_USE_AITER off", dict(use_aiter=False, aiter=no_aiter)),
            ("switch off", dict(enabled=False, aiter=no_aiter)),
            ("TP4 front shape", dict(weight=_shaped(7552, 7168))),
            ("fp16 front", dict(weight=_shaped(6016, 7168, torch.float16))),
            ("gfx942", dict(aiter=_fake_aiter(gfx="gfx942"))),
        ):
            with self.subTest(name):
                self.assertIsNone(self._build(**kwargs))

    def test_fails_at_load_on_an_aiter_without_tgemm_or_its_rows(self):
        without_tgemm = {**_fake_aiter(), "aiter.tuned_gemm": None}
        with self.assertRaises(ImportError):
            self._build(aiter=without_tgemm)
        with self.assertRaises(RuntimeError):
            self._build(aiter=_fake_aiter(has_rows=False))


if __name__ == "__main__":
    unittest.main()
