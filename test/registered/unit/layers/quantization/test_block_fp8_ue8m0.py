"""Regression tests for block-FP8 UE8M0 activation-scale propagation.

A checkpoint may declare ``scale_fmt="ue8m0"`` for block-FP8 weights.
Fp8LinearMethod converts that metadata into ``act_scale_ue8m0=True``.

For 128-wide K blocks, the backend dispatcher must preserve that semantic
through the CUTLASS path and its Triton fallback. The row-padded activation
quantizer used by CUTLASS must also quantize with power-of-two UE8M0 scales.
"""

import functools
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sglang.kernels.ops.quantization.fp8_kernel as fp8_kernel
import sglang.srt.layers.quantization.fp8_utils as fp8_utils
from sglang.srt.layers.quantization.fp8_utils import (
    Fp8GemmRunnerBackend,
    cutlass_w8a8_block_fp8_linear_with_fallback,
    dispatch_w8a8_block_fp8_linear,
)
from sglang.test.test_utils import CustomTestCase

BLOCK_SIZE = [128, 128]


class TestBlockFp8Ue8m0(CustomTestCase):
    def test_cutlass_dispatch_preserves_ue8m0_flag(self):
        platform = SimpleNamespace(is_sm120=True)

        with (
            patch.object(
                fp8_utils,
                "FP8_GEMM_RUNNER_BACKEND",
                Fp8GemmRunnerBackend("cutlass"),
            ),
            patch.object(fp8_utils, "get_platform", return_value=platform),
        ):
            fn = dispatch_w8a8_block_fp8_linear(
                BLOCK_SIZE,
                act_scale_ue8m0=True,
            )

        self.assertIsInstance(fn, functools.partial)
        self.assertIs(
            fn.func,
            cutlass_w8a8_block_fp8_linear_with_fallback,
        )
        self.assertEqual(
            fn.keywords,
            {"act_scale_ue8m0": True},
        )

    def test_cutlass_passes_ue8m0_to_row_padded_quantizer(self):
        m, n, k = 2, 128, 128

        x = torch.zeros((m, k), dtype=torch.bfloat16)
        weight = torch.zeros((n, k), dtype=torch.bfloat16)
        weight_scale = torch.ones((1, 1), dtype=torch.float32)

        q_input = torch.zeros(
            (4, k),
            dtype=torch.float8_e4m3fn,
        )
        x_scale = torch.ones((4, 1), dtype=torch.float32)

        quant_spy = MagicMock(return_value=(q_input, x_scale))
        gemm_spy = MagicMock(return_value=torch.zeros((4, n), dtype=torch.bfloat16))

        with (
            patch.object(
                fp8_utils,
                "sglang_per_token_group_quant_fp8_row_padded",
                quant_spy,
            ),
            patch.object(
                fp8_utils,
                "fp8_blockwise_scaled_mm",
                gemm_spy,
            ),
        ):
            out = cutlass_w8a8_block_fp8_linear_with_fallback(
                input=x,
                weight=weight,
                block_size=BLOCK_SIZE,
                weight_scale=weight_scale,
                act_scale_ue8m0=True,
            )

        self.assertEqual(tuple(out.shape), (m, n))

        args, kwargs = quant_spy.call_args
        self.assertEqual(args[1], 128)
        self.assertTrue(kwargs["scale_ue8m0"])

    def test_cutlass_triton_fallback_preserves_ue8m0_flag(self):
        m, n, k = 2, 96, 128

        x = torch.zeros((m, k), dtype=torch.bfloat16)
        weight = torch.zeros((n, k), dtype=torch.bfloat16)
        weight_scale = torch.ones((1, 1), dtype=torch.float32)

        triton_spy = MagicMock(return_value=torch.zeros((m, n), dtype=torch.bfloat16))

        with patch.object(
            fp8_utils,
            "triton_w8a8_block_fp8_linear",
            triton_spy,
        ):
            cutlass_w8a8_block_fp8_linear_with_fallback(
                input=x,
                weight=weight,
                block_size=BLOCK_SIZE,
                weight_scale=weight_scale,
                act_scale_ue8m0=True,
            )

        self.assertEqual(
            triton_spy.call_args.kwargs["act_scale_ue8m0"],
            True,
        )

    def test_row_padded_quantizer_supports_ue8m0(self):
        # Use two scale groups so row-major vs column-major layout is visible.
        x = torch.zeros((3, 256), dtype=torch.bfloat16)

        observed = {}

        def fake_quant_kernel(
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            *,
            scale_ue8m0,
            fuse_silu_and_mul,
            masked_m,
        ):
            observed["scale_ue8m0"] = scale_ue8m0
            observed["scale_contiguous"] = x_s.is_contiguous()

            x_q.zero_()
            x_s.fill_(2.0)

        with patch.object(
            fp8_kernel,
            "_run_per_token_group_quant_8bit_kernel",
            side_effect=fake_quant_kernel,
        ):
            q, scale = fp8_kernel.sglang_per_token_group_quant_fp8_row_padded(
                x,
                group_size=128,
                scale_ue8m0=True,
            )

        self.assertTrue(observed["scale_ue8m0"])

        # UE8M0's fp32 quant kernel must write to a row-major temporary;
        # CUTLASS still receives the usual column-major padded scale buffer.
        self.assertTrue(observed["scale_contiguous"])
        self.assertEqual(tuple(q.shape), (4, 256))
        self.assertEqual(tuple(scale.shape), (4, 2))
        self.assertEqual(scale.stride(0), 1)

        torch.testing.assert_close(
            scale[:3],
            torch.full((3, 2), 2.0),
        )
        torch.testing.assert_close(
            scale[3:],
            torch.zeros((1, 2)),
        )


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=3)
