"""Producer-level validation of the gfx95 bpreshuffle fp8-scale no-copy path.

The CPU tests in ``test_fp8_bpreshuffle_scale.py`` only prove that the stride
reinterpret recovers a *fabricated* ``transpose_scale=True`` layout. They cannot
catch the real AITER producers ignoring or mis-implementing ``transpose_scale``.
These tests invoke the two producers the optimization actually routes through --
``fused_clamp_act_mul`` (MoE down) and ``fused_flatten_fp8_group_quant`` (MLA
o_proj) -- and prove, on real gfx95 kernels, that the two producer paths are
equivalent:

    transpose_scale=True  + view_aiter_fused_rms_transposed_fp8_scale   (the optimized path)
    transpose_scale=False + materialize_bpreshuffle_fp8_scale   (the row-major path)

For M(tokens) >= 2 they must agree bit-for-bit on the quantized output and on the
scale *values* after relayout, with the no-copy path landing on the bpreshuffle
``(1, M)`` column-major stride and sharing the producer's storage. M == 1 is the
materialize-only fallback (``emit_transposed_bpreshuffle_scale`` gates the
transposed emit on M >= 2); we pin that its materialized scale is well-formed.

Requires a gfx95 (MI35X) GPU with aiter; skips otherwise.
"""

import unittest

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    emit_transposed_bpreshuffle_scale,
    materialize_bpreshuffle_fp8_scale,
    view_aiter_fused_rms_transposed_fp8_scale,
)
from sglang.srt.utils.common import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_GROUP_SIZE = 128


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(),
    "bpreshuffle fp8-scale no-copy is a gfx95 (MI35X) + aiter optimization",
)
class TestBpreshuffleProducerScaleNoCopy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from aiter import dtypes  # noqa: F401
            from aiter.ops.triton.fused_fp8_quant import (  # noqa: F401
                fused_flatten_fp8_group_quant,
            )
            from aiter.ops.triton.fusions.fused_clamp_act_mul import (  # noqa: F401
                fused_clamp_act_mul,
            )
        except Exception as err:  # pragma: no cover - env-dependent
            raise unittest.SkipTest(f"aiter producers unavailable: {err}")
        cls.device = "cuda"  # torch maps "cuda" onto the ROCm HIP device

    def setUp(self):
        torch.manual_seed(0)

    # --- producer adapters: run the producer and normalize to (q, scale) ------

    def _run_fused_clamp_act_mul(self, m, transpose_scale):
        from aiter import dtypes
        from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul

        inter = 4 * _GROUP_SIZE  # G = 4 groups
        gate_up = torch.randn(m, 2 * inter, device=self.device, dtype=torch.bfloat16)
        q, scale = fused_clamp_act_mul(
            gate_up,
            swiglu_limit=7.0,
            activation="silu",
            dtype_quant=dtypes.fp8,
            transpose_scale=transpose_scale,
        )
        return q, scale, gate_up

    def _run_fused_flatten_fp8_group_quant(self, m, transpose_scale):
        from aiter.ops.triton.fused_fp8_quant import fused_flatten_fp8_group_quant

        heads, dim = 8, _GROUP_SIZE  # heads*dim = 1024 -> G = 8 groups
        buf = torch.randn(m, heads, dim, device=self.device, dtype=torch.bfloat16)
        out = fused_flatten_fp8_group_quant(
            buf,
            group_size=_GROUP_SIZE,
            dtype_quant=torch.float8_e4m3fn,
            transpose_scale=transpose_scale,
        )
        return out[0], out[1], buf

    def _assert_producer_paths_equivalent(self, run_producer, name):
        for m in (1, 2, 8, 16):
            with self.subTest(producer=name, m=m):
                # Row-major path (transpose_scale=False) + materialize. This is
                # the path M == 1 takes in production, so it must always be valid.
                torch.manual_seed(m)
                q_f, s_f, _ = run_producer(m, transpose_scale=False)
                mat = materialize_bpreshuffle_fp8_scale(s_f)
                self.assertEqual(mat.dim(), 2)
                self.assertEqual(mat.shape[0], m)

                if not emit_transposed_bpreshuffle_scale(m, on_bpreshuffle_gfx95=True):
                    # M == 1: the transposed emit is skipped by design. At M == 1 the
                    # [1, G] row-major and [G, 1] column-major byte orders coincide,
                    # so materialize is a no-op that keeps the natural (G, 1) stride
                    # (NOT the (1, M) column-major stride taken for M >= 2) while
                    # sharing storage; the scale values must survive intact.
                    self.assertEqual(m, 1)
                    self.assertEqual(mat.stride(), (s_f.shape[1], 1))  # (G, 1)
                    self.assertTrue(torch.equal(mat, s_f))
                    continue

                # M >= 2: materialize lands on the bpreshuffle (1, M) column-major
                # stride.
                self.assertEqual(mat.stride(), (1, m))

                # Optimized path: same input, transpose_scale=True + no-copy view.
                torch.manual_seed(m)
                q_t, s_t, _ = run_producer(m, transpose_scale=True)
                nocopy = view_aiter_fused_rms_transposed_fp8_scale(s_t)

                # Quantized output is layout-independent: transpose_scale only
                # changes the *scale* storage, never the quantized bytes.
                self.assertEqual(
                    q_t.dtype, q_f.dtype, "quant dtype differs between paths"
                )
                self.assertTrue(
                    torch.equal(q_t.view(torch.uint8), q_f.view(torch.uint8)),
                    "quantized output differs between transpose_scale paths",
                )

                # Scale values match the row-major + materialize path exactly...
                self.assertEqual(nocopy.shape, mat.shape)
                self.assertTrue(
                    torch.equal(nocopy, mat),
                    "no-copy scale values differ from materialized",
                )
                # ...on the bpreshuffle (1, M) stride, with no allocation.
                self.assertEqual(nocopy.stride(), (1, m))
                self.assertEqual(nocopy.data_ptr(), s_t.data_ptr())

    def test_fused_clamp_act_mul_producer_paths_equivalent(self):
        self._assert_producer_paths_equivalent(
            self._run_fused_clamp_act_mul, "fused_clamp_act_mul"
        )

    def test_fused_flatten_fp8_group_quant_producer_paths_equivalent(self):
        self._assert_producer_paths_equivalent(
            self._run_fused_flatten_fp8_group_quant, "fused_flatten_fp8_group_quant"
        )


if __name__ == "__main__":
    unittest.main()
