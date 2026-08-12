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


if __name__ == "__main__":
    unittest.main()
