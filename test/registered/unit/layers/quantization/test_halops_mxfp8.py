"""Contract tests for the HALOps MXFP8 integration.

These run without the optional ``halops`` wheel: the SfAtom weight-scale layout
and the arguments the dense-linear wrapper hands to HALOps are checked with the
extension mocked out.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8_utils, halops_mxfp8_linear
from sglang.srt.layers.quantization.halops_mxfp8 import (
    HALOPS_MODE,
    swizzle_scale_to_sfatom,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SF_ATOM_ROWS = 128
SCALE_GROUP = 32


def _sfatom_reference_k4(scale: torch.Tensor) -> torch.Tensor:
    """CUTLASS SfAtom layout for a four-column scale (K == 128).

    ``out[(a * 32 + r2) * 4 + r1] == src[a * 128 + r1 * 32 + r2]``: each 128-row
    atom splits into four 32-row strips that interleave down the output rows.
    """
    m, k_groups = scale.shape
    assert k_groups == 4, "this derivation covers one 128x4 atom group"

    out = torch.empty_like(scale)
    for atom in range(m // SF_ATOM_ROWS):
        for r1 in range(4):
            for r2 in range(32):
                out[(atom * 32 + r2) * 4 + r1] = scale[
                    atom * SF_ATOM_ROWS + r1 * 32 + r2
                ]
    return out


class TestSfAtomScaleLayout(CustomTestCase):
    def test_matches_cutlass_128x4_atom_order(self):
        for m in (SF_ATOM_ROWS, 2 * SF_ATOM_ROWS, 4 * SF_ATOM_ROWS):
            with self.subTest(m=m):
                scale = torch.arange(m * 4, dtype=torch.int32).reshape(m, 4)
                out = swizzle_scale_to_sfatom(scale)
                self.assertEqual(out.shape, scale.shape)
                self.assertTrue(torch.equal(out, _sfatom_reference_k4(scale)))

    def test_preserves_values_across_scale_groups(self):
        scale = torch.randint(0, 2**30, (2 * SF_ATOM_ROWS, 16), dtype=torch.int64)
        out = swizzle_scale_to_sfatom(scale)
        self.assertEqual(out.shape, scale.shape)
        self.assertEqual(
            sorted(out.flatten().tolist()), sorted(scale.flatten().tolist())
        )


def _patch_halops(*, group_wise_quant, dense_gemm):
    return mock.patch.object(
        halops_mxfp8_linear,
        "load_halops_gemm",
        return_value=(dense_gemm, group_wise_quant),
    )


def _patch_sm100():
    return mock.patch.object(
        halops_mxfp8_linear,
        "get_platform",
        return_value=SimpleNamespace(is_sm100=True),
    )


class TestLinearWiring(CustomTestCase):
    def test_forwards_mode_scale_and_bias_dtype(self):
        m, n, k = 256, 512, 1024
        weight = torch.zeros((n, k), dtype=torch.float8_e4m3fn)
        weight_scale = torch.zeros((n, k // SCALE_GROUP), dtype=torch.uint8)
        sfa = torch.zeros((m, k // SCALE_GROUP), dtype=torch.uint8)
        quantized = torch.zeros((m, k), dtype=torch.float8_e4m3fn)
        group_wise_quant = mock.Mock(return_value=(quantized, sfa))
        dense_gemm = mock.Mock()

        x = torch.zeros((2, m // 2, k), dtype=torch.float16)
        bias = torch.zeros(n, dtype=torch.float16)

        with (
            _patch_halops(group_wise_quant=group_wise_quant, dense_gemm=dense_gemm),
            _patch_sm100(),
        ):
            out = halops_mxfp8_linear.halops_w8a8_mxfp8_linear_with_fallback(
                input=x, weight=weight, weight_scale=weight_scale, bias=bias
            )

        quant_input = group_wise_quant.call_args.args[0]
        self.assertEqual(quant_input.shape, (m, k))
        self.assertTrue(quant_input.is_contiguous())
        self.assertEqual(
            group_wise_quant.call_args.kwargs,
            {"mode": HALOPS_MODE, "use_swizzle": True},
        )

        kwargs = dense_gemm.call_args.kwargs
        self.assertIs(kwargs["input"], quantized)
        self.assertIs(kwargs["weight"], weight)
        self.assertIs(kwargs["sfa"], sfa)
        self.assertIs(kwargs["sfb"], weight_scale)
        self.assertEqual(kwargs["mode"], HALOPS_MODE)
        self.assertEqual(kwargs["sm_count"], -1)
        self.assertIsNone(kwargs["alpha"])
        # dense_gemm types the bias as its bf16 output element type.
        self.assertEqual(kwargs["bias"].dtype, torch.bfloat16)

        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(out.shape, x.shape[:-1] + (n,))

    def test_rejects_prequantized_activation_scale(self):
        weight = torch.zeros((512, 1024), dtype=torch.float8_e4m3fn)
        weight_scale = torch.zeros((512, 1024 // SCALE_GROUP), dtype=torch.uint8)
        x = torch.zeros((64, 1024), dtype=torch.bfloat16)

        with self.assertRaises(NotImplementedError):
            halops_mxfp8_linear.halops_w8a8_mxfp8_linear_with_fallback(
                input=x,
                weight=weight,
                weight_scale=weight_scale,
                input_scale=torch.ones(64, dtype=torch.float32),
            )


class TestFlashinferFallback(CustomTestCase):
    @staticmethod
    def _inputs():
        m, n, k = 64, 512, 1024
        return (
            torch.zeros((m, k), dtype=torch.float32),
            torch.zeros((n, k), dtype=torch.float8_e4m3fn),
            torch.zeros((n, k // SCALE_GROUP), dtype=torch.uint8),
        )

    def test_downcasts_fp32_and_restores_caller_dtype(self):
        x, weight, weight_scale = self._inputs()
        interleaved = torch.zeros_like(weight_scale)
        fake_flashinfer = types.ModuleType("flashinfer")
        fake_flashinfer.block_scale_interleave = mock.Mock(return_value=interleaved)
        linear = mock.Mock(
            return_value=torch.zeros(
                (x.shape[0], weight.shape[0]), dtype=torch.bfloat16
            )
        )

        with (
            mock.patch.dict(sys.modules, {"flashinfer": fake_flashinfer}),
            mock.patch.object(fp8_utils, "flashinfer_mxfp8_blockscaled_linear", linear),
            # The fallback must not need HALOps to be importable.
            mock.patch.object(halops_mxfp8_linear, "load_halops_gemm") as load,
        ):
            out = halops_mxfp8_linear.halops_w8a8_mxfp8_linear_with_fallback(
                input=x,
                weight=weight,
                weight_scale=weight_scale,
                weight_scale_fallback=weight_scale,
            )

        load.assert_not_called()
        self.assertEqual(linear.call_args.kwargs["input"].dtype, torch.bfloat16)
        self.assertIs(linear.call_args.kwargs["weight_scale"], interleaved)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.shape, (x.shape[0], weight.shape[0]))

    def test_requires_a_canonical_scale_for_the_fallback(self):
        x, weight, weight_scale = self._inputs()

        with self.assertRaisesRegex(RuntimeError, "flashinfer_cutlass"):
            halops_mxfp8_linear.halops_w8a8_mxfp8_linear_with_fallback(
                input=x,
                weight=weight,
                weight_scale=weight_scale,
                weight_scale_fallback=None,
            )


if __name__ == "__main__":
    unittest.main()
