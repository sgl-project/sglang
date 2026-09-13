"""Real-path validation of the dense w8a8 bpreshuffle fp8-scale no-copy on gfx95.

The CPU tests in ``test_fp8_bpreshuffle_scale.py`` only prove the stride formula
against a fabricated layout. They cannot catch ``aiter_per1x128_quant(
transpose_scale=True)`` emitting the wrong layout, the ``emit_bpreshuffle_scale``
gating being wrong, or an integration mismatch with the CK bpreshuffle GEMM.

These tests exercise the real kernels ``aiter_w8a8_block_fp8_linear`` routes
through and prove the optimized (``transpose_scale=True`` + no-copy view) path is
equivalent to the original (``transpose_scale=False`` + materialize copy) path:

- ``test_quant_producer_scale_equivalence`` -- at the quant-producer level:
  identical quantized bytes, identical scale *values* after relayout, the
  bpreshuffle ``(1, M)`` column-major stride, and zero-copy storage sharing.
- ``test_dense_linear_paths_bit_exact`` -- end-to-end through
  ``aiter_w8a8_block_fp8_linear``: the new path (as shipped) vs the old path
  (forced by patching the quant to row-major + the relayout to materialize) must
  produce **bit-identical** GEMM output. Covers M == 1 (materialize fallback).

Requires a gfx95 (MI35X) GPU with aiter and ROCm >= 7.2 (bpreshuffle); skips
otherwise.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8_utils import (
    materialize_bpreshuffle_fp8_scale,
    view_aiter_fused_rms_transposed_fp8_scale,
)
from sglang.srt.utils.common import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd-mi35x")

# N, K chosen off the tuned-Triton list so aiter_w8a8_block_fp8_linear takes the
# CK bpreshuffle GEMM path (use_triton == False) -- the path this PR optimizes.
_N, _K = 512, 256
_BLOCK = [128, 128]


@unittest.skipUnless(
    is_hip() and is_gfx95_supported() and fp8_utils._use_aiter_bpreshuffle_gfx95,
    "dense bpreshuffle scale no-copy is a gfx95 (MI35X) + aiter + ROCm>=7.2 path",
)
class TestDenseBpreshuffleScaleNoCopy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # These module globals only exist when aiter/gfx95 imports succeeded.
        for attr in ("aiter", "aiter_per1x128_quant"):
            if not hasattr(fp8_utils, attr):
                raise unittest.SkipTest(f"fp8_utils.{attr} unavailable (no aiter)")
        cls.device = "cuda"  # torch maps "cuda" onto the ROCm HIP device

    def setUp(self):
        torch.manual_seed(0)

    def _rand_input(self, m):
        return torch.randn(m, _K, device=self.device, dtype=torch.bfloat16)

    # ---------------------------------------------------------- quant producer

    def test_quant_producer_scale_equivalence(self):
        fp8 = fp8_utils.aiter.dtypes.fp8
        for m in (1, 2, 8, 16):
            with self.subTest(m=m):
                x = self._rand_input(m)

                # Original path: row-major emit + materialize.
                q_f, s_f = fp8_utils.aiter_per1x128_quant(
                    x, quant_dtype=fp8, transpose_scale=False
                )
                mat = materialize_bpreshuffle_fp8_scale(s_f)

                # M == 1 stays on the materialize path in production (>= 2 gate).
                # There materialize is a no-op: the [1, G] row-major and [G, 1]
                # column-major byte orders coincide, so it keeps the natural (G, 1)
                # stride (NOT the (1, M) column-major stride taken for M >= 2) and
                # shares storage; the scale values must survive intact.
                if m < 2:
                    self.assertEqual(mat.stride(), (s_f.shape[1], 1))  # (G, 1)
                    self.assertTrue(torch.equal(mat, s_f))
                    continue

                self.assertEqual(mat.stride(), (1, m))  # bpreshuffle column-major

                # Optimized path: transposed emit + zero-copy view.
                q_t, s_t = fp8_utils.aiter_per1x128_quant(
                    x, quant_dtype=fp8, transpose_scale=True
                )
                nocopy = view_aiter_fused_rms_transposed_fp8_scale(s_t)

                # Quantized bytes are layout-independent.
                self.assertTrue(
                    torch.equal(q_t.view(torch.uint8), q_f.view(torch.uint8)),
                    "quantized output differs between transpose_scale paths",
                )
                # Scale values match the materialized copy, on the (1, M) stride,
                # with no allocation (view over the producer's buffer).
                self.assertEqual(nocopy.shape, mat.shape)
                self.assertTrue(torch.equal(nocopy, mat))
                self.assertEqual(nocopy.stride(), (1, m))
                self.assertEqual(nocopy.data_ptr(), s_t.data_ptr())

    # ---------------------------------------------------------- end-to-end GEMM

    def _make_weight(self):
        finfo = torch.finfo(torch.float8_e4m3fn)
        w = (torch.rand(_N, _K, device=self.device, dtype=torch.float32) - 0.5) * 2
        weight = (w * finfo.max).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
        weight_scale = (
            torch.rand(
                _N // _BLOCK[0],
                _K // _BLOCK[1],
                device=self.device,
                dtype=torch.float32,
            )
            * 1e-2
            + 1e-3
        )
        return weight, weight_scale

    def test_dense_linear_paths_bit_exact(self):
        weight, weight_scale = self._make_weight()
        real_quant = fp8_utils.aiter_per1x128_quant

        def _row_major_quant(inp, **kwargs):
            # Force the pre-optimization behavior: emit the scale row-major.
            kwargs["transpose_scale"] = False
            return real_quant(inp, **kwargs)

        for m in (1, 2, 8, 16):
            with self.subTest(m=m):
                x = self._rand_input(m)

                # New path, exactly as shipped. Also pin that it really takes the
                # CK bpreshuffle GEMM (use_triton == False) -- the path this PR
                # optimizes -- so a future tuned-shape-list change can't silently
                # route this coverage through Triton and void the equivalence check.
                with (
                    mock.patch.object(
                        fp8_utils,
                        "gemm_a8w8_blockscale_bpreshuffle",
                        wraps=fp8_utils.gemm_a8w8_blockscale_bpreshuffle,
                    ) as spy_bpreshuffle,
                    mock.patch.object(
                        fp8_utils,
                        "triton_gemm_a8w8_blockscale",
                        wraps=fp8_utils.triton_gemm_a8w8_blockscale,
                    ) as spy_triton,
                ):
                    out_new = fp8_utils.aiter_w8a8_block_fp8_linear(
                        x, weight, _BLOCK, weight_scale
                    )
                spy_bpreshuffle.assert_called_once()
                spy_triton.assert_not_called()

                # Old path: row-major quant + materialize relayout. Patching both
                # the quant flag and the relayout helper reconstructs the original
                # `transpose_scale=False` + `materialize_bpreshuffle_fp8_scale`
                # branch through the same public function and downstream GEMM.
                with (
                    mock.patch.object(
                        fp8_utils, "aiter_per1x128_quant", _row_major_quant
                    ),
                    mock.patch.object(
                        fp8_utils,
                        "view_aiter_fused_rms_transposed_fp8_scale",
                        materialize_bpreshuffle_fp8_scale,
                    ),
                ):
                    out_old = fp8_utils.aiter_w8a8_block_fp8_linear(
                        x, weight, _BLOCK, weight_scale
                    )

                self.assertEqual(out_new.shape, out_old.shape)
                self.assertTrue(
                    torch.equal(out_new, out_old),
                    f"dense linear output differs between scale-layout paths (M={m})",
                )


if __name__ == "__main__":
    unittest.main()
