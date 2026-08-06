import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    materialize_bpreshuffle_fp8_scale,
    materialize_bpreshuffle_fp8_scale_tuple,
    view_aiter_fused_rms_transposed_fp8_scale,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _simulate_transpose_scale_emit(values: torch.Tensor) -> torch.Tensor:
    """Model the scale a quant kernel returns when called with
    ``transpose_scale=True``: the per-group scale is written directly in
    column-major (``[num_groups, tokens]``) byte order, exposed as a ``[M, G]``
    tensor. We reproduce that by laying the column-major bytes into contiguous
    storage and reinterpreting it as ``[M, G]`` -- the logical row-major view is
    scrambled, but the *storage* holds exactly the bytes the no-copy stride
    reinterpret is meant to recover."""
    m, g = values.shape
    colmajor_bytes = values.t().contiguous()  # [G, M], storage == col-major of values
    return colmajor_bytes.view(m, g)  # [M, G] over the same (unchanged) storage


def _as_strided_nocopy(x_scale: torch.Tensor) -> torch.Tensor:
    """The exact fresh-quant no-copy reinterpret from ``aiter_w8a8_block_fp8_linear``."""
    return torch.as_strided(x_scale, x_scale.shape, (1, x_scale.shape[0]))


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


class TestBpreshuffleScaleFreshQuantNoCopy(CustomTestCase):
    """The dense w8a8 fresh-quant path asks the quant kernel for the scale in
    bpreshuffle byte-order (``transpose_scale=True``) and reinterprets its strides
    with ``torch.as_strided`` instead of relaying it out with
    ``materialize_bpreshuffle_fp8_scale`` (a ``.t().contiguous().t()`` copy). These
    pin the PR's core claim: the reinterpret is bit-identical to the copy path for
    M>=2, and allocates nothing."""

    def test_as_strided_matches_materialize(self):
        for m, g in ((3, 4), (2, 2), (8, 5), (16, 128)):
            with self.subTest(m=m, g=g):
                values = torch.arange(m * g, dtype=torch.float32).reshape(m, g)
                emitted = _simulate_transpose_scale_emit(values)

                nocopy = _as_strided_nocopy(emitted)
                materialized = materialize_bpreshuffle_fp8_scale(values)

                self.assertTrue(torch.equal(nocopy, materialized))
                self.assertEqual(nocopy.shape, values.shape)
                self.assertEqual(nocopy.stride(), (1, m))
                self.assertEqual(nocopy.stride(), materialized.stride())
                self.assertTrue(nocopy.t().is_contiguous())

    def test_as_strided_shares_storage_no_allocation(self):
        values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        emitted = _simulate_transpose_scale_emit(values)

        nocopy = _as_strided_nocopy(emitted)

        # The reinterpret is a view over the producer's buffer -- no new storage.
        self.assertEqual(nocopy.data_ptr(), emitted.data_ptr())
        # ...unlike the materialize path it replaces.
        materialized = materialize_bpreshuffle_fp8_scale(values)
        self.assertNotEqual(materialized.data_ptr(), values.data_ptr())


if __name__ == "__main__":
    unittest.main()
