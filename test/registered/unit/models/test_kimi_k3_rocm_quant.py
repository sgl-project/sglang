"""Unit tests for srt/models/kimi_k3_rocm_quant and the K3 weight-merge guard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention, _is_unquantized_mergeable
from sglang.srt.models.kimi_k3_rocm_quant import (
    _k3_channel_fp8_to_bf16,
    k3_kda_inproj_fp8_tail_rows,
    k3_merge_kda_inproj_fp8,
)
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


def _quark_fp8_linear(out_features: int, in_features: int = 16):
    scheme = QuarkW8A8Fp8(
        {"qscheme": "per_channel"}, {"qscheme": "per_channel", "is_dynamic": True}
    )
    scheme.out_dtype = torch.bfloat16
    weight, scale, _ = _per_channel_fp8(out_features, in_features)
    return SimpleNamespace(
        weight=weight,
        weight_scale=scale.squeeze(1),
        scheme=scheme,
        quant_method=mock.Mock(spec=QuarkLinearMethod),
    )


def _dequant(linear) -> torch.Tensor:
    return linear.weight.float() * linear.weight_scale.view(-1, 1)


def _kda_inproj_linears(b_proj=None) -> dict:
    """[q,k,v,g] = 32 rows plus the tail k3_kda_inproj_fp8_tail_rows reserves."""
    mods = dict(
        f_a_proj=_quark_fp8_linear(8),
        b_proj=_quark_fp8_linear(3) if b_proj is None else b_proj,
        f_b_proj=_quark_fp8_linear(16, in_features=8),
    )
    tail = k3_kda_inproj_fp8_tail_rows(wide_rows=32, **mods)
    return dict(qkvg_proj=_quark_fp8_linear(32 + tail), qkvg_sizes=[24, 8], **mods)


class TestMergeKdaInprojFp8(CustomTestCase):
    """The merged buffer has to carry every projection's channel scales; a raw
    row-cat of FP8 payloads would silently drop them."""

    def test_tail_pads_the_merged_width_to_64(self):
        # 32 + 8 + 3 = 43 rows; N % 64 == 0 keeps AITER's preshuffled FP8 GEMM.
        mods = _kda_inproj_linears()
        self.assertEqual(mods["qkvg_proj"].weight.shape[0], 64)

    def test_merged_rows_keep_every_channel_scale(self):
        mods = _kda_inproj_linears()
        expected = torch.cat(
            [
                _dequant(mods["qkvg_proj"])[:32],
                _dequant(mods["f_a_proj"]),
                _dequant(mods["b_proj"]),
            ]
        )

        sizes, f_b = k3_merge_kda_inproj_fp8(**mods)

        # Checkpoint layout, [out, in] and [out], for the loader to process.
        merged = _dequant(mods["qkvg_proj"])
        torch.testing.assert_close(merged[:43], expected)
        self.assertEqual(merged[43:].abs().max().item(), 0.0)
        self.assertEqual(sizes, [24, 8, 8, 3, 21])
        torch.testing.assert_close(f_b, _dequant(mods["f_b_proj"]).to(torch.bfloat16))

    def test_merge_fills_the_reserved_rows_in_place(self):
        """Swapping in a new, slightly wider buffer would free the wide weight
        and strand caching-allocator segments on every KDA layer."""
        mods = _kda_inproj_linears()
        weight, scale = mods["qkvg_proj"].weight, mods["qkvg_proj"].weight_scale
        ptrs = (weight.data_ptr(), scale.data_ptr())

        k3_merge_kda_inproj_fp8(**mods)

        self.assertIs(mods["qkvg_proj"].weight, weight)
        self.assertEqual((weight.data_ptr(), scale.data_ptr()), ptrs)

    def test_no_tail_for_a_projection_that_is_not_quark_fp8(self):
        bf16_b_proj = SimpleNamespace(
            weight=torch.zeros(3, 16, dtype=torch.bfloat16), quant_method=None
        )
        mods = _kda_inproj_linears(b_proj=bf16_b_proj)
        self.assertEqual(mods["qkvg_proj"].weight.shape[0], 32)
        self.assertIsNone(k3_merge_kda_inproj_fp8(**mods))

    def test_projection_without_a_channel_scale_skips_the_merge(self):
        """A BF16 or block-FP8 Kimi-K3 checkpoint has no per-channel weight_scale;
        loading it must skip the FP8 merge instead of raising."""
        mods = {
            name: SimpleNamespace(
                weight=torch.zeros(rows, cols, dtype=torch.bfloat16), quant_method=None
            )
            for name, rows, cols in (
                ("qkvg_proj", 32, 16),
                ("f_a_proj", 8, 16),
                ("b_proj", 3, 16),
                ("f_b_proj", 16, 8),
            )
        }
        self.assertIsNone(k3_merge_kda_inproj_fp8(qkvg_sizes=[24, 8], **mods))

    def test_env_disables_the_tail(self):
        with envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ.override(False):
            mods = _kda_inproj_linears()
        self.assertEqual(mods["qkvg_proj"].weight.shape[0], 32)
        self.assertIsNone(k3_merge_kda_inproj_fp8(**mods))

    @staticmethod
    def _kda_attention() -> torch.nn.Module:
        mods = _kda_inproj_linears()
        attn = torch.nn.Module()
        attn.__dict__.update(
            fused_qkvg_proj=mods["qkvg_proj"],
            split_sizes=mods["qkvg_sizes"],
            f_a_proj=mods["f_a_proj"],
            b_proj=mods["b_proj"],
            f_b_proj=mods["f_b_proj"],
            _qkvgbfa_in_qkvg_proj=False,
            _qkvgbfa_sizes=None,
            _bfa_f_b_w=None,
            _derived_weight_cache_error=None,
        )
        return attn

    def test_merged_layer_rejects_online_weight_updates(self):
        """f_a/b rows moved into fused_qkvg_proj and the cached BF16 f_b would
        not follow a reload, so the updater must refuse before writing."""
        attn = self._kda_attention()
        model = torch.nn.Sequential(attn)
        self.assertIsNone(_unsupported_derived_weight_cache_error(model))

        self.assertTrue(KimiK3DeltaAttention._merge_kda_inproj_fp8_hip(attn))

        self.assertEqual(attn.fused_qkvg_proj.weight.shape[0], 64)
        self.assertIn(
            "SGLANG_ROCM_K3_FUSE_KDA_INPROJ=0",
            _unsupported_derived_weight_cache_error(model),
        )

    def test_rerun_keeps_the_merge(self):
        """A second post_load_weights must keep the FP8 merge, not fall back to the
        BF16 merge over weights that no longer have the checkpoint layout."""
        attn = self._kda_attention()
        self.assertTrue(KimiK3DeltaAttention._merge_kda_inproj_fp8_hip(attn))
        weight, sizes = attn.fused_qkvg_proj.weight, attn._qkvgbfa_sizes

        self.assertTrue(KimiK3DeltaAttention._merge_kda_inproj_fp8_hip(attn))

        self.assertIs(attn.fused_qkvg_proj.weight, weight)
        self.assertEqual(attn._qkvgbfa_sizes, sizes)


if __name__ == "__main__":
    unittest.main()
