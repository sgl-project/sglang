"""CPU checks for the block-fp8-as-MXFP8 per-layer admission gate.

The gate must not be looser than the kernel the dispatch picks: b12x needs
K % 128 == 0, so a K that is a multiple of 32 but not of 128 has to keep the
Triton block kernel instead of being converted and then silently served by
CUTLASS. See issue #49789.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.quantization import fp8, fp8_utils
from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

SM120 = 120
# 9 blocks of 32: admitted by the K % 32 condition, unusable by b12x.
K_32_ONLY = 288
K_128_ALIGNED = 512
N = 96

CUTLASS = Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS
NOT_B12X_ROUTES = (
    Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL,
    Mxfp8DenseGemmBackend.GFX95_DOT_SCALED,
    Mxfp8DenseGemmBackend.GFX95_MXFP8_NATIVE,
    Mxfp8DenseGemmBackend.XPU_SCALED_MM,
)


def _fake_flashinfer(b12x_supported: bool):
    return SimpleNamespace(
        is_backend_supported=lambda name, sm: name == "b12x" and b12x_supported
    )


def _patches(b12x_supported: bool, sm: int = SM120):
    return (
        mock.patch.object(
            fp8_utils,
            "_raw_flashinfer_mm_mxfp8",
            _fake_flashinfer(b12x_supported),
            create=True,
        ),
        mock.patch.object(fp8_utils, "get_device_sm", return_value=sm),
    )


class TestBlockFp8Mxfp8CanServeK(CustomTestCase):
    def _can_serve(self, backend, k: int, b12x_supported: bool) -> bool:
        flashinfer, device_sm = _patches(b12x_supported)
        with flashinfer, device_sm:
            return fp8_utils.block_fp8_mxfp8_can_serve_k(backend, k)

    def test_b12x_route_requires_a_full_bk128_tile(self):
        self.assertFalse(self._can_serve(CUTLASS, K_32_ONLY, True))
        self.assertTrue(self._can_serve(CUTLASS, K_128_ALIGNED, True))

    def test_cutlass_without_b12x_is_not_restricted(self):
        self.assertTrue(self._can_serve(CUTLASS, K_32_ONLY, False))

    def test_other_backends_keep_their_existing_admission(self):
        for backend in NOT_B12X_ROUTES:
            with self.subTest(backend=backend.value):
                self.assertTrue(self._can_serve(backend, K_32_ONLY, True))

    def test_gate_and_dispatch_agree_on_the_same_condition(self):
        """The only guarantee the fix needs: the gate tightens exactly where the
        dispatch would hand a K%128!=0 layer to CUTLASS."""
        for b12x_supported in (True, False):
            flashinfer, device_sm = _patches(b12x_supported)
            with flashinfer, device_sm:
                takes_b12x_route = (
                    fp8_utils.dispatch_block_fp8_mxfp8_linear(CUTLASS)
                    is fp8_utils._b12x_or_cutlass_mxfp8_linear
                )
                self.assertEqual(takes_b12x_route, b12x_supported)
                self.assertEqual(
                    fp8_utils.block_fp8_mxfp8_can_serve_k(CUTLASS, K_32_ONLY),
                    not takes_b12x_route,
                )


class _Method:
    """Stand-in for Fp8LinearMethod carrying only what the gate reads."""

    def __init__(self, backend):
        self.mxfp8_dense_backend = backend
        self.weight_block_size = [32, 32]
        self.processed = []

    def _process_mxfp8_linear_weight_scale(self, layer, scale_u8=None):
        self.processed.append(scale_u8)
        layer.weight_scale_inv_swizzled = scale_u8


class TestPrepareBlockFp8AsMxfp8Admission(CustomTestCase):
    def setUp(self):
        fp8._BLOCK_FP8_MXFP8_K_SKIPPED.clear()

    def tearDown(self):
        fp8._BLOCK_FP8_MXFP8_K_SKIPPED.clear()

    def _layer(self, k: int, keep_plain: bool = False):
        weight = torch.zeros((N, k), dtype=torch.float8_e4m3fn)
        scale = torch.zeros((N // 32, k // 32), dtype=torch.uint8)
        layer = SimpleNamespace(
            weight=weight,
            weight_scale_inv=scale,
            block_fp8_mxfp8_ready=None,
        )
        if keep_plain:
            layer.keep_plain_weight_layout = True
        return layer

    def _prepare(self, layer, backend=CUTLASS, b12x_supported: bool = True):
        method = _Method(backend)
        flashinfer, device_sm = _patches(b12x_supported)
        with (
            flashinfer,
            device_sm,
            mock.patch.object(
                fp8, "block_fp8_scale_to_mxfp8_e8m0", return_value="SWIZZLED"
            ) as convert,
        ):
            fp8.Fp8LinearMethod._prepare_block_fp8_as_mxfp8(method, layer)
        return method, convert

    def test_k_that_is_only_32_aligned_stays_on_the_block_kernel(self):
        layer = self._layer(K_32_ONLY)
        method, convert = self._prepare(layer)
        self.assertFalse(layer.block_fp8_mxfp8_ready)
        convert.assert_not_called()
        self.assertEqual(method.processed, [])
        # The Triton fallback reads the checkpoint scales, so they must be intact.
        self.assertIsNone(getattr(layer, "weight_scale_inv_swizzled", None))
        self.assertEqual(
            tuple(layer.weight_scale_inv.shape), (N // 32, K_32_ONLY // 32)
        )

    def test_128_aligned_k_still_takes_the_view(self):
        layer = self._layer(K_128_ALIGNED)
        method, convert = self._prepare(layer)
        self.assertTrue(layer.block_fp8_mxfp8_ready)
        convert.assert_called_once()
        self.assertEqual(method.processed, ["SWIZZLED"])

    def test_refused_k_is_warned_about_once_per_shape(self):
        with self.assertLogs(level="WARNING") as logs:
            self._prepare(self._layer(K_32_ONLY))
            self._prepare(self._layer(K_32_ONLY))
        self.assertEqual(
            sum("stays on the Triton block kernel" in line for line in logs.output), 1
        )
        self.assertIn((N, K_32_ONLY), fp8._BLOCK_FP8_MXFP8_K_SKIPPED)

    def test_other_backends_are_not_affected_by_the_new_gate(self):
        for backend in NOT_B12X_ROUTES:
            with self.subTest(backend=backend.value):
                layer = self._layer(K_32_ONLY)
                method, convert = self._prepare(layer, backend=backend)
                self.assertTrue(layer.block_fp8_mxfp8_ready)
                convert.assert_called_once()

    def test_plain_weight_layout_still_short_circuits(self):
        layer = self._layer(K_128_ALIGNED, keep_plain=True)
        method, convert = self._prepare(layer)
        self.assertFalse(layer.block_fp8_mxfp8_ready)
        convert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
