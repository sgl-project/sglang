"""SM120 small-M per-tensor FP8 fast path in apply_fp8_linear_bmm_flashinfer."""

import unittest
from unittest import mock

import torch
from flashinfer import bmm_fp8 as flashinfer_bmm_fp8_raw

from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear_bmm_flashinfer
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max

SHAPES = ((16384, 5120), (5120, 6144), (14336, 5120))
M_VALUES = (4, 8, 16, 20, 24, 32, 60)


def _reference(x, w_kn, x_scale, w_scale):
    q, xs = static_quant_fp8(x.view(-1, x.shape[-1]), x_scale, repeat_scale=False)
    return (q.float() * xs) @ (w_kn.float() * w_scale)


def _cublas_linear(x, w_kn, w_scale, x_scale):
    """Baseline via raw flashinfer cuBLAS, bypassing the SGLang per-tensor dispatch."""
    q, xs = static_quant_fp8(x.view(-1, x.shape[-1]), x_scale, repeat_scale=False)
    m, n = q.shape[0], w_kn.shape[1]
    return flashinfer_bmm_fp8_raw(
        q.unsqueeze(0),
        w_kn.unsqueeze(0),
        xs.reshape(1),
        w_scale.reshape(1),
        x.dtype,
        backend="cublas",
    ).view(m, n)


@unittest.skipUnless(is_sm120_supported(), "requires SM120 (>= 12.0)")
class TestFp8PertensorDispatch(CustomTestCase):
    def test_production_path_matches_cublas(self):
        from sglang.kernels.ops.gemm import fp8_pertensor_gemm as pertensor

        expected_calls = sum(
            pertensor.is_profitable(m, n, k) for n, k in SHAPES for m in M_VALUES
        )
        # Without this the test still passes when the fast path never runs.
        self.assertLess(0, expected_calls)
        self.assertLess(expected_calls, len(SHAPES) * len(M_VALUES))

        gen = torch.Generator(device="cuda").manual_seed(1)
        x_scale = torch.tensor(0.01, device="cuda", dtype=torch.float32)
        w_scale = torch.tensor(0.01, device="cuda", dtype=torch.float32)

        with mock.patch.object(
            pertensor,
            "fp8_pertensor_scaled_mm",
            wraps=pertensor.fp8_pertensor_scaled_mm,
        ) as gemm_spy:
            for n, k in SHAPES:
                wb = torch.randn(
                    n, k, device="cuda", dtype=torch.bfloat16, generator=gen
                ).mul_(0.1)
                w_kn = wb.clamp_(-FP8_MAX, FP8_MAX).to(FP8).t()
                for m in M_VALUES:
                    x = torch.randn(
                        m, k, device="cuda", dtype=torch.bfloat16, generator=gen
                    ).mul_(0.1)
                    ref = _reference(x, w_kn, x_scale, w_scale)
                    out_cublas = _cublas_linear(x, w_kn, w_scale, x_scale)
                    out = apply_fp8_linear_bmm_flashinfer(x, w_kn, w_scale, x_scale)

                    self.assertEqual(out.shape, (m, n))
                    torch.testing.assert_close(
                        out.float(), ref, rtol=0.05, atol=0.05, msg=f"{m=} {n=} {k=}"
                    )
                    torch.testing.assert_close(
                        out.float(),
                        out_cublas.float(),
                        rtol=0.02,
                        atol=0.02,
                        msg=f"{m=} {n=} {k=}",
                    )

        self.assertEqual(gemm_spy.call_count, expected_calls)


if __name__ == "__main__":
    unittest.main()
