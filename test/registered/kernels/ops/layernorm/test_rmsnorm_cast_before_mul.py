"""RMSNorm that rounds the normalized activation before the weight multiply (HF RMSNorm order)."""

import unittest

import torch
from sgl_kernel import fused_add_rmsnorm, rmsnorm

from sglang.kernels.ops.layernorm.rmsnorm_cast_before_mul import (
    rmsnorm_cast_before_mul,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")

# The fused fp32 sum differs from torch's in the last bit, so a few outputs move by up to two
# rounding steps; measured on B60 and A100 (also CUDA's fused kernel): <= 3.6e-5 and 1.85 steps.
MAX_MISMATCH_FRACTION = 1e-4
MAX_DIFF_IN_ROUNDING_STEPS = 2


def accelerator_device() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return None


def hf_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    normalized = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    return weight * normalized.to(x.dtype)


def assert_hf_rounding(
    test: CustomTestCase, output: torch.Tensor, expected: torch.Tensor
) -> None:
    mismatch_fraction = (output != expected).float().mean().item()
    rounding_step = expected.float().abs() * torch.finfo(expected.dtype).eps
    allowed_diff = MAX_DIFF_IN_ROUNDING_STEPS * rounding_step
    test.assertLessEqual(mismatch_fraction, MAX_MISMATCH_FRACTION)
    test.assertTrue(
        bool(((output.float() - expected.float()).abs() <= allowed_diff).all())
    )


@unittest.skipIf(accelerator_device() is None, "needs a CUDA or XPU device")
class TestRMSNormCastBeforeMul(CustomTestCase):
    def test_matches_hf_rounding_order(self):
        device = accelerator_device()
        # Hidden-size and per-head norms of SenseNova-U1, plus a non-power-of-two width.
        shapes = [(4096, 4096), (4096 * 32, 128), (300, 4096), (33, 3584)]
        for (rows, dim), dtype in [
            (s, d) for s in shapes for d in (torch.bfloat16, torch.float16)
        ]:
            with self.subTest(rows=rows, dim=dim, dtype=dtype):
                torch.manual_seed(rows + dim)
                x = torch.randn(rows, dim, device=device, dtype=dtype) * 3
                weight = torch.empty(dim, device=device, dtype=dtype).normal_(1.0, 0.2)
                output = rmsnorm_cast_before_mul(x, weight, 1e-6)
                assert_hf_rounding(self, output, hf_rmsnorm(x, weight, 1e-6))

    def test_strided_rows_from_chunked_heads(self):
        """SenseNova norms each half of q/k from chunk(2, dim=-1): rows are strided, not contiguous."""
        device = accelerator_device()
        heads = torch.randn(1, 4096, 32, 128, device=device, dtype=torch.bfloat16) * 3
        weight = torch.empty(64, device=device, dtype=torch.bfloat16).normal_(1.0, 0.2)
        for half in heads.chunk(2, dim=-1):
            output = rmsnorm_cast_before_mul(half, weight, 1e-6)
            self.assertEqual(output.shape, half.shape)
            assert_hf_rounding(self, output, hf_rmsnorm(half, weight, 1e-6))

    @unittest.skipIf(
        accelerator_device() is None or torch.accelerator.device_count() < 2,
        reason="needs two accelerator devices",
    )
    def test_runs_on_the_input_device_queue(self):
        """The kernel once launched on the current device's queue, so on a split model it read inputs still being written on the other device."""
        device_type = accelerator_device().type
        torch.get_device_module(device_type).set_device(0)
        second_device = torch.device(device_type, 1)
        a = torch.randn(512, 32768, device=second_device, dtype=torch.bfloat16)
        b = torch.randn(32768, 4096, device=second_device, dtype=torch.bfloat16) / 181
        weight = torch.empty(4096, device=second_device, dtype=torch.bfloat16)
        weight.normal_(1.0, 0.2)
        # Each trial queues matmuls on the input device right before the norm; the old
        # wrapper read the unfinished input in 9 of 10 trials (B60), so ten trials fail it.
        for _ in range(10):
            x = a @ b
            for _ in range(3):
                x = (x @ b[:4096]) / 64
            output = rmsnorm_cast_before_mul(x, weight, 1e-6)
            assert_hf_rounding(self, output, hf_rmsnorm(x, weight, 1e-6))

    @unittest.skipUnless(torch.xpu.is_available(), "needs an XPU device")
    def test_xpu_forward_keeps_hf_rounding(self):
        """XPU RMSNorm with cast_x_before_out_mul once multiplied the weight before rounding (~25 % of outputs off)."""
        norm = RMSNorm(4096, eps=1e-6, cast_x_before_out_mul=True).to(
            "xpu", torch.bfloat16
        )
        norm.weight.data.normal_(1.0, 0.2)
        x = torch.randn(512, 4096, device="xpu", dtype=torch.bfloat16) * 3
        assert_hf_rounding(
            self, norm.forward_xpu(x), hf_rmsnorm(x, norm.weight.data, 1e-6)
        )
        # The residual form took sgl_kernel fused_add_rmsnorm, which also multiplies first.
        residual = torch.randn_like(x)
        output, new_residual = norm.forward_xpu(x.clone(), residual.clone())
        expected, expected_residual = norm.forward_native(x, residual)
        torch.testing.assert_close(new_residual, expected_residual, rtol=0, atol=0)
        assert_hf_rounding(self, output, expected)

    def test_residual_matches_native_and_leaves_inputs_untouched(self):
        """With a residual the kernel must return forward_native's (output, residual) and not write its inputs."""
        device = accelerator_device()
        norm = RMSNorm(4096, eps=1e-6, cast_x_before_out_mul=True).to(
            device, torch.bfloat16
        )
        norm.weight.data.normal_(1.0, 0.2)
        x = torch.randn(512, 4096, device=device, dtype=torch.bfloat16) * 3
        residual = torch.randn_like(x)
        post = torch.randn_like(x)
        for post_residual_addition in (None, post):
            with self.subTest(
                post_residual_addition=post_residual_addition is not None
            ):
                inputs = (x.clone(), residual.clone(), post.clone())
                output, new_residual = rmsnorm_cast_before_mul(
                    x,
                    norm.weight.data,
                    1e-6,
                    residual=residual,
                    post_residual_addition=post_residual_addition,
                )
                expected, expected_residual = norm.forward_native(
                    x, residual, post_residual_addition
                )
                # The residual sum is exact fp32 in both, so only the norm's reduction differs.
                torch.testing.assert_close(
                    new_residual, expected_residual, rtol=0, atol=0
                )
                assert_hf_rounding(self, output, expected)
                for before, after in zip(inputs, (x, residual, post)):
                    torch.testing.assert_close(after, before, rtol=0, atol=0)

    @unittest.skipUnless(torch.xpu.is_available(), "needs an XPU device")
    def test_xpu_default_norms_still_use_sgl_kernel(self):
        """Norms without cast_x_before_out_mul must keep the upstream sgl_kernel results on XPU."""
        norm = RMSNorm(4096, eps=1e-6).to("xpu", torch.bfloat16)
        norm.weight.data.normal_(1.0, 0.2)
        x = torch.randn(512, 4096, device="xpu", dtype=torch.bfloat16) * 3
        residual = torch.randn_like(x)
        torch.testing.assert_close(
            norm.forward_xpu(x.clone()),
            rmsnorm(x.clone(), norm.weight.data, 1e-6),
            rtol=0,
            atol=0,
        )
        expected_x, expected_residual = x.clone(), residual.clone()
        fused_add_rmsnorm(expected_x, expected_residual, norm.weight.data, 1e-6)
        output, output_residual = norm.forward_xpu(x.clone(), residual.clone())
        torch.testing.assert_close(
            (output, output_residual), (expected_x, expected_residual), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
