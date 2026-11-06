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

from sglang.srt.layers.quantization import (
    fp8_utils,
    halops_mxfp8,
    halops_mxfp8_linear,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend
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


class TestOfflineCheckpointScales(CustomTestCase):
    """Load-time handling of a checkpoint that is already quantized.

    The weights and scales come from disk, so nothing may re-quantize them, and
    the SfAtom layout must be derived from the scale the checkpoint carries.
    """

    @staticmethod
    def _layer(*, n, k, scale):
        return SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.zeros((n, k), dtype=torch.float8_e4m3fn), requires_grad=False
            ),
            weight_scale_inv=torch.nn.Parameter(scale, requires_grad=False),
        )

    @staticmethod
    def _method(**attrs):
        # Bypass __init__: these cases drive the load-time scale prep only.
        method = Fp8LinearMethod.__new__(Fp8LinearMethod)
        for name, value in attrs.items():
            setattr(method, name, value)
        return method

    def test_serialized_mxfp8_keeps_the_checkpoint_scale(self):
        n, k = 2 * SF_ATOM_ROWS, 128
        scale = torch.randint(120, 130, (n, k // SCALE_GROUP), dtype=torch.uint8)
        layer = self._layer(n=n, k=k, scale=scale)
        method = self._method(
            use_mxfp8=True, mxfp8_dense_backend=Mxfp8DenseGemmBackend.HALOPS
        )

        with mock.patch.object(halops_mxfp8, "quantize_weight") as quantize_weight:
            method._process_mxfp8_linear_weight_scale(layer)

        # The checkpoint's own scale is relaid out, never re-derived from weights.
        quantize_weight.assert_not_called()
        self.assertTrue(torch.equal(layer.weight_scale_inv.data, scale))
        self.assertTrue(
            torch.equal(layer.weight_scale_inv_halops, _sfatom_reference_k4(scale))
        )

    def test_second_load_refreshes_the_derived_scale(self):
        # update_weights_from_disk replaces weight_scale_inv and re-runs this
        # step without recapturing CUDA graphs, so the cached SfAtom bytes must
        # follow the new scale while staying in the storage the graphs captured.
        n, k = SF_ATOM_ROWS, 4 * SCALE_GROUP
        scale = torch.full((n, k // SCALE_GROUP), 127, dtype=torch.uint8)
        layer = self._layer(n=n, k=k, scale=scale)
        method = self._method(
            use_mxfp8=True, mxfp8_dense_backend=Mxfp8DenseGemmBackend.HALOPS
        )

        method._process_mxfp8_linear_weight_scale(layer)
        first = layer.weight_scale_inv_halops
        self.assertTrue(torch.equal(first.data, _sfatom_reference_k4(scale)))

        layer.weight_scale_inv.data.fill_(128)
        method._process_mxfp8_linear_weight_scale(layer)

        # UE8M0 byte 128 is a 2x multiplier where 127 was 1x.
        self.assertTrue(torch.equal(first.data, _sfatom_reference_k4(scale)))
        self.assertEqual(layer.weight_scale_inv_halops.data_ptr(), first.data_ptr())

    def test_unaligned_n_keeps_only_the_checkpoint_scale(self):
        n, k = SF_ATOM_ROWS + 64, 128
        scale = torch.randint(120, 130, (n, k // SCALE_GROUP), dtype=torch.uint8)
        layer = self._layer(n=n, k=k, scale=scale)
        method = self._method(
            use_mxfp8=True, mxfp8_dense_backend=Mxfp8DenseGemmBackend.HALOPS
        )

        method._process_mxfp8_linear_weight_scale(layer)

        # No SfAtom layout exists; apply's FlashInfer fallback reads
        # weight_scale_inv, which is still the checkpoint's own scale.
        self.assertIsNone(layer.weight_scale_inv_halops)
        self.assertTrue(torch.equal(layer.weight_scale_inv.data, scale))


class TestOnlineQuantizationBuffers(CustomTestCase):
    """A BF16 checkpoint quantized at load is requantized on every load."""

    @staticmethod
    def _method():
        method = Fp8LinearMethod.__new__(Fp8LinearMethod)
        method.use_mxfp8 = True
        method.mxfp8_dense_backend = Mxfp8DenseGemmBackend.HALOPS
        return method

    @staticmethod
    def _quantized(value, scale_value, *, n, k):
        scale = torch.full((n, k // SCALE_GROUP), scale_value, dtype=torch.uint8)
        qweight = torch.full((n, k), value, dtype=torch.float8_e4m3fn)
        return qweight, swizzle_scale_to_sfatom(scale), scale

    def test_requantization_writes_through_the_captured_buffers(self):
        """update_weights_from_disk does not recapture CUDA graphs, so a second
        load must overwrite what the first load handed to dense_gemm."""
        n, k = SF_ATOM_ROWS, 128
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.zeros((n, k), dtype=torch.bfloat16), requires_grad=False
            ),
            # First load has no weight_scale_inv yet; the quantizer registers it.
            register_parameter=lambda name, param: setattr(layer, name, param),
        )
        method = self._method()

        with mock.patch.object(halops_mxfp8, "quantize_weight") as quantize_weight:
            quantize_weight.side_effect = [
                self._quantized(1.0, 127, n=n, k=k),
                self._quantized(2.0, 128, n=n, k=k),
            ]
            method._quantize_mxfp8_weights(layer)
            captured = {
                name: getattr(layer, name).data_ptr()
                for name in ("weight", "weight_scale_inv", "weight_scale_inv_halops")
            }
            method._quantize_mxfp8_weights(layer)

        self.assertEqual(quantize_weight.call_count, 2)
        for name, data_ptr in captured.items():
            self.assertEqual(getattr(layer, name).data_ptr(), data_ptr, name)
        self.assertTrue(torch.equal(layer.weight.data.float(), torch.full((n, k), 2.0)))
        expected_scale = torch.full((n, k // SCALE_GROUP), 128, dtype=torch.uint8)
        self.assertTrue(torch.equal(layer.weight_scale_inv.data, expected_scale))
        self.assertTrue(
            torch.equal(
                layer.weight_scale_inv_halops.data,
                swizzle_scale_to_sfatom(expected_scale),
            )
        )


if __name__ == "__main__":
    unittest.main()
