import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    emit_transposed_bpreshuffle_scale,
    materialize_bpreshuffle_fp8_scale,
    materialize_bpreshuffle_fp8_scale_tuple,
    view_aiter_fused_rms_transposed_fp8_scale,
    view_aiter_fused_rms_transposed_fp8_scale_tuple,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _simulate_transpose_scale_emit(values: torch.Tensor) -> torch.Tensor:
    """Model the tensor a quant kernel returns when called with
    ``transpose_scale=True``: the per-group scale is written directly in
    column-major (``[num_groups, tokens]``) byte order, exposed as a ``[M, G]``
    tensor. We reproduce that by laying the column-major bytes into contiguous
    storage and reinterpreting it as ``[M, G]`` -- the logical row-major view is
    scrambled, but the *storage* holds exactly the bytes the no-copy stride
    reinterpret is meant to recover."""
    m, g = values.shape
    colmajor_bytes = values.t().contiguous()  # [G, M], storage == col-major of values
    return colmajor_bytes.view(m, g)  # [M, G] over the same (unchanged) storage


class TestBpreshuffleScaleMaterialization(CustomTestCase):
    def test_materializes_transposed_physical_storage(self):
        scale = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        materialized = materialize_bpreshuffle_fp8_scale(scale)

        self.assertTrue(torch.equal(materialized, scale))
        self.assertEqual(materialized.shape, scale.shape)
        self.assertEqual(materialized.stride(), (1, scale.shape[0]))
        self.assertTrue(materialized.t().is_contiguous())

    def test_materialization_is_idempotent_for_bpreshuffle_layout(self):
        scale = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        materialized = materialize_bpreshuffle_fp8_scale(scale)

        rematerialized = materialize_bpreshuffle_fp8_scale(materialized)

        self.assertTrue(torch.equal(rematerialized, scale))
        self.assertEqual(rematerialized.stride(), materialized.stride())
        self.assertEqual(rematerialized.data_ptr(), materialized.data_ptr())

    def test_repairs_aiter_scale_before_downstream_layout_handling(self):
        """AITER-transposed scale bytes must retain their logical indexing.

        AITER ``transpose_scale=True`` returns transposed physical storage with
        row-major-looking metadata. Treating that metadata as logical layout
        permutes the scales during CK materialization.
        """
        logical_scale = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        aiter_scale = logical_scale.t().contiguous().view(logical_scale.shape)

        repaired = view_aiter_fused_rms_transposed_fp8_scale(aiter_scale)
        materialized = materialize_bpreshuffle_fp8_scale(repaired)
        renormalized = view_aiter_fused_rms_transposed_fp8_scale(repaired)

        self.assertTrue(torch.equal(repaired, logical_scale))
        self.assertTrue(torch.equal(materialized, logical_scale))
        self.assertTrue(torch.equal(renormalized, logical_scale))
        self.assertEqual(repaired.stride(), (1, logical_scale.shape[0]))
        self.assertEqual(repaired.data_ptr(), aiter_scale.data_ptr())
        self.assertEqual(materialized.data_ptr(), aiter_scale.data_ptr())
        self.assertEqual(renormalized.stride(), repaired.stride())
        self.assertEqual(renormalized.data_ptr(), aiter_scale.data_ptr())

    def test_deepseek_v4_repairs_fused_rms_scale_at_producer(self):
        """DeepSeek-V4 must repair fused-RMS scale metadata before CK consumes it."""
        from sglang.srt.models import deepseek_v4

        q_input = torch.ones((3, 1024), dtype=torch.float32)
        x_bf16 = torch.ones((3, 1024), dtype=torch.bfloat16)
        logical_scale = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        aiter_scale = logical_scale.t().contiguous().view(logical_scale.shape)
        fused_output = ((q_input, aiter_scale), x_bf16, None, None)

        with (
            patch.object(
                deepseek_v4,
                "fused_rms_fp8_group_quant",
                return_value=fused_output,
                create=True,
            ),
            patch.object(deepseek_v4, "_use_aiter_bpreshuffle_gfx95", True),
        ):
            x_quant, x_unquantized = deepseek_v4._fused_rmsnorm_fp8_quant(
                q_input, torch.ones(1024), 1e-6
            )

        self.assertIs(x_quant[0], q_input)
        self.assertIs(x_unquantized, x_bf16)
        self.assertTrue(torch.equal(x_quant[1], logical_scale))
        self.assertEqual(x_quant[1].stride(), (1, logical_scale.shape[0]))
        self.assertEqual(x_quant[1].data_ptr(), aiter_scale.data_ptr())

    def test_tuple_helper_keeps_extra_tuple_payload(self):
        q_input = torch.ones((3, 8), dtype=torch.float32)
        scale = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        bf16_side = torch.ones((3, 8), dtype=torch.bfloat16)

        q_out, scale_out, bf16_out = materialize_bpreshuffle_fp8_scale_tuple(
            (q_input, scale, bf16_side)
        )

        self.assertIs(q_out, q_input)
        self.assertIs(bf16_out, bf16_side)
        self.assertTrue(torch.equal(scale_out, scale))
        self.assertEqual(scale_out.stride(), (1, scale.shape[0]))


class TestBpreshuffleScaleNoCopy(CustomTestCase):
    """The producer sites emit the scale with ``transpose_scale=True`` and then
    reinterpret its strides instead of relaying it out with a copy. These pin the
    contract that the zero-copy reinterpret is bit-identical to the materialize
    (``.t().contiguous().t()``) path while sharing the producer's storage."""

    def test_nocopy_matches_materialized_layout(self):
        for m, g in ((3, 4), (2, 2), (8, 5), (16, 128)):
            with self.subTest(m=m, g=g):
                values = torch.arange(m * g, dtype=torch.float32).reshape(m, g)
                emitted = _simulate_transpose_scale_emit(values)

                nocopy = view_aiter_fused_rms_transposed_fp8_scale(emitted)
                # Row-major producer path (transpose_scale=False + materialize).
                materialized = materialize_bpreshuffle_fp8_scale(values)

                self.assertTrue(torch.equal(nocopy, materialized))
                self.assertEqual(nocopy.shape, values.shape)
                self.assertEqual(nocopy.stride(), (1, m))
                self.assertTrue(nocopy.t().is_contiguous())

    def test_nocopy_shares_storage_no_allocation(self):
        values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        emitted = _simulate_transpose_scale_emit(values)

        nocopy = view_aiter_fused_rms_transposed_fp8_scale(emitted)

        # The whole point of the optimization: reinterpret, do not copy.
        self.assertEqual(nocopy.data_ptr(), emitted.data_ptr())
        # ...whereas the materialize path it replaces does allocate.
        materialized = materialize_bpreshuffle_fp8_scale(values)
        self.assertNotEqual(materialized.data_ptr(), values.data_ptr())

    def test_nocopy_passthrough_for_non_2d_scale(self):
        for scale in (
            torch.arange(5, dtype=torch.float32),  # 1-D (per-tensor scale)
            torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),  # 3-D
        ):
            with self.subTest(dim=scale.dim()):
                self.assertIs(view_aiter_fused_rms_transposed_fp8_scale(scale), scale)

    def test_tuple_helper_reinterprets_only_the_scale_slot(self):
        q_input = torch.ones((3, 8), dtype=torch.float8_e4m3fn)
        values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        emitted = _simulate_transpose_scale_emit(values)
        bf16_side = torch.ones((3, 8), dtype=torch.bfloat16)

        q_out, scale_out, bf16_out = view_aiter_fused_rms_transposed_fp8_scale_tuple(
            (q_input, emitted, bf16_side)
        )

        self.assertIs(q_out, q_input)
        self.assertIs(bf16_out, bf16_side)
        self.assertTrue(
            torch.equal(scale_out, materialize_bpreshuffle_fp8_scale(values))
        )
        self.assertEqual(scale_out.stride(), (1, values.shape[0]))
        self.assertEqual(scale_out.data_ptr(), emitted.data_ptr())


class TestEmitTransposedBpreshuffleScaleGate(CustomTestCase):
    """Pins the producer emit-gate shared by the MoE-down and MLA o_proj sites:
    the transposed zero-copy path is taken only on gfx95 bpreshuffle and only for
    M(tokens) >= 2; M == 1 must fall back to the materialize path. Guards the
    ``>= 2`` boundary against being widened to `M >= 1` (which would send a
    degenerate single-token scale down the stride-swap path)."""

    def test_gate_false_off_gfx95_regardless_of_m(self):
        for m in (1, 2, 8):
            with self.subTest(m=m):
                self.assertFalse(
                    emit_transposed_bpreshuffle_scale(m, on_bpreshuffle_gfx95=False)
                )

    def test_gate_requires_m_ge_2_on_gfx95(self):
        # M == 1 -> materialize fallback; M >= 2 -> transposed zero-copy path.
        self.assertFalse(
            emit_transposed_bpreshuffle_scale(1, on_bpreshuffle_gfx95=True)
        )
        for m in (2, 3, 16):
            with self.subTest(m=m):
                self.assertTrue(
                    emit_transposed_bpreshuffle_scale(m, on_bpreshuffle_gfx95=True)
                )

    def test_m1_materialize_path_produces_correct_values_and_layout(self):
        # The path M == 1 actually takes: transpose_scale=False output relaid out
        # by materialize. A single-token scale must survive with its values intact
        # and the bpreshuffle (1, M) == (1, 1) column-major stride.
        self.assertFalse(
            emit_transposed_bpreshuffle_scale(1, on_bpreshuffle_gfx95=True)
        )
        scale = torch.arange(4, dtype=torch.float32).reshape(1, 4)  # [M=1, G=4]
        materialized = materialize_bpreshuffle_fp8_scale(scale)
        self.assertTrue(torch.equal(materialized, scale))
        self.assertEqual(materialized.shape, (1, 4))
        self.assertEqual(materialized.stride(), (1, 1))
        self.assertTrue(materialized.t().is_contiguous())


if __name__ == "__main__":
    unittest.main()
