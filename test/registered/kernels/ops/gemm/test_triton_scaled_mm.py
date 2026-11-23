import unittest
from typing import Optional

import torch
import torch.testing

from sglang.kernels.ops.gemm.fp8_kernel import triton_scaled_mm
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.utils.device import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=13, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=12, suite="base-b-test-1-gpu-small-amd")


def torch_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation using float32 for stability"""
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a.to(torch.float32) * out * scale_b.to(torch.float32).T
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(out_dtype)


class TestScaledMM(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not (torch.cuda.is_available() or torch.xpu.is_available()):
            raise unittest.SkipTest("No CUDA or XPU device available")
        cls._device = get_device()
        torch.set_default_device(cls._device)

    def _make_inputs(self, M, K, N, in_dtype):
        if in_dtype == torch.int8:
            a = torch.randint(-8, 8, (M, K), dtype=in_dtype, device=self._device)
            b = torch.randint(-8, 8, (K, N), dtype=in_dtype, device=self._device)
        else:  # fp8
            a = torch.clamp(
                0.1 * torch.randn((M, K), dtype=torch.float16, device=self._device),
                -0.3,
                0.3,
            ).to(in_dtype)
            b = torch.clamp(
                0.1 * torch.randn((K, N), dtype=torch.float16, device=self._device),
                -0.3,
                0.3,
            ).to(in_dtype)
        return a, b

    def test_basic_cases(self):
        """Test core functionality with reduced precision requirements"""
        test_configs = [
            (32, 32, 32, torch.int8, torch.float16, False),
            (17, 64, 96, torch.int8, torch.float16, False),
            (64, 64, 64, torch.int8, torch.float16, True),
        ]

        try:
            torch.tensor([1.0], dtype=torch.float8_e4m3fn, device=self._device)
            test_configs.append((32, 32, 32, torch.float8_e4m3fn, torch.float16, False))
            test_configs.append((17, 64, 96, torch.float8_e4m3fn, torch.float16, False))
        except:
            print("FP8 not supported, skipping")

        for M, K, N, in_dtype, out_dtype, with_bias in test_configs:
            with self.subTest(M=M, K=K, N=N, dtype=in_dtype, bias=with_bias):
                print(f"Currently testing with in_dtype: {in_dtype}")
                torch.manual_seed(42)

                input, weight = self._make_inputs(M, K, N, in_dtype)
                scale_a = 0.1 + 0.05 * torch.rand(
                    (M, 1), dtype=torch.float32, device=self._device
                )
                scale_b = 0.1 + 0.05 * torch.rand(
                    (N, 1), dtype=torch.float32, device=self._device
                )
                bias = (
                    0.01 * torch.randn((M, N), dtype=out_dtype, device=self._device)
                    if with_bias
                    else None
                )

                triton_out = triton_scaled_mm(
                    input, weight, scale_a, scale_b, out_dtype, bias
                )
                ref_out = torch_scaled_mm(
                    input, weight, scale_a, scale_b, out_dtype, bias
                )

                # Use relaxed tolerances
                rtol = 0.15 if in_dtype == torch.int8 else 0.25
                atol = 0.1 if in_dtype == torch.int8 else 0.15

                torch.testing.assert_close(triton_out, ref_out, rtol=rtol, atol=atol)

                scale_b_row = scale_b.t().contiguous()
                triton_out_row_scale = triton_scaled_mm(
                    input, weight, scale_a, scale_b_row, out_dtype, bias
                )
                torch.testing.assert_close(
                    triton_out_row_scale, ref_out, rtol=rtol, atol=atol
                )

    def test_int8_matches_int32_reference(self):
        """W8A8 INT8 layout: per-token activations, per-channel weights, K x N weight view."""
        mantissa_bits = {torch.bfloat16: 7, torch.float16: 10}
        for M, N, K in [
            (1, 1152, 896),
            (7, 896, 896),
            (33, 4864, 896),
            (300, 896, 4864),
        ]:
            for out_dtype in (torch.bfloat16, torch.float16):
                for with_bias in (False, True):
                    with self.subTest(
                        M=M, N=N, K=K, out_dtype=out_dtype, bias=with_bias
                    ):
                        torch.manual_seed(0)
                        x = torch.randn((M, K), dtype=out_dtype, device=self._device)
                        w = torch.randint(
                            -127, 128, (N, K), dtype=torch.int8, device=self._device
                        )
                        w_scale = 1e-4 + 1e-3 * torch.rand(
                            (N, 1), dtype=torch.float32, device=self._device
                        )
                        bias = (
                            4 * torch.randn((N,), dtype=out_dtype, device=self._device)
                            if with_bias
                            else None
                        )
                        x_q, x_scale = per_token_quant_int8(x)

                        out = triton_scaled_mm(
                            x_q, w.t(), x_scale, w_scale, out_dtype, bias
                        ).double()

                        # fp64 holds the INT32 accumulator exactly (|acc| < 2^53).
                        ref = (x_q.double() @ w.double().t()) * x_scale.double()
                        ref = ref * w_scale.double().t()
                        magnitude = ref.abs()
                        if bias is not None:
                            ref = ref + bias.double()
                            magnitude = magnitude + bias.double().abs()
                        # Half an ulp of out_dtype at the larger of out and ref,
                        # never below half the smallest subnormal.
                        exponent = torch.frexp(
                            torch.maximum(out.abs(), ref.abs())
                        ).exponent
                        half_ulp = torch.ldexp(
                            torch.full_like(ref, 0.5),
                            exponent - 1 - mantissa_bits[out_dtype],
                        ).clamp_min(
                            torch.finfo(out_dtype).smallest_normal
                            * 2.0 ** -mantissa_bits[out_dtype]
                            / 2
                        )
                        # Slack for the fp32 epilogue (int32 to fp32, two scales, bias).
                        bound = half_ulp + magnitude * 2.0**-20
                        excess = (out - ref).abs() - bound
                        self.assertLessEqual(
                            excess.max().item(),
                            0.0,
                            f"{(excess > 0).float().mean().item():.4f} of outputs "
                            "are more than 1/2 ulp from the reference",
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
