import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=20, stage="nightly", runner_config="1-gpu-large")


def sglang_jit_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    cast_x_before_out_mul: bool = False,
) -> None:
    from sglang.kernels.ops.layernorm.norm import fused_add_rmsnorm

    fused_add_rmsnorm(
        input, residual, weight, eps, cast_x_before_out_mul=cast_x_before_out_mul
    )


def flashinfer_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    from flashinfer.norm import fused_add_rmsnorm

    fused_add_rmsnorm(input, residual, weight, eps=eps)


def forward_native_hf_reference(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    sum_fp32 = x.to(torch.float32) + residual.to(torch.float32)
    residual_out = sum_fp32.to(x.dtype)
    variance = sum_fp32.pow(2).mean(-1, keepdim=True)
    out = w * (sum_fp32 * torch.rsqrt(variance + eps)).to(x.dtype)
    return out, residual_out


def forward_scaled_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    input_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = (x.float() * input_scale).to(x.dtype)
    residual_out = (residual.float() + scaled.float()).to(x.dtype)
    residual_fp32 = residual_out.float()
    variance = residual_fp32.pow(2).mean(-1, keepdim=True)
    out = (residual_fp32 * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)
    return out, residual_out


BS_LIST = [2**n for n in range(0, 14)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
HIDDEN_SIZE_LIST = [512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
FUSED_ADD_RMSNORM_CASES = get_ci_test_range(
    list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST)),
    [
        (1, 512),
        (18, 4096),
        (38, 4096),
        (39, 4096),
        (39, 5120),
        (39, 8192),
        (44, 8192),
        (89, 4096),
    ],
)
DEVICE = "cuda"
DTYPE = torch.bfloat16
EPS = torch.finfo(torch.bfloat16).eps
SCALED_EPS = 1e-5


@pytest.mark.parametrize(
    "batch_size,hidden_size,cast_x_before_out_mul",
    [(bs, hs, cast) for bs, hs in FUSED_ADD_RMSNORM_CASES for cast in [False, True]],
)
def test_fused_add_rmsnorm(
    batch_size: int, hidden_size: int, cast_x_before_out_mul: bool
) -> None:
    torch.manual_seed(0)
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
    residual = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)

    input_sglang = input.clone()
    residual_sglang = residual.clone()
    sglang_jit_fused_add_rmsnorm(
        input_sglang,
        residual_sglang,
        weight,
        EPS,
        cast_x_before_out_mul=cast_x_before_out_mul,
    )

    if cast_x_before_out_mul:
        out_ref, residual_ref = forward_native_hf_reference(
            input, residual, weight, EPS
        )
    else:
        input_ref = input.clone()
        residual_ref_buf = residual.clone()
        flashinfer_fused_add_rmsnorm(input_ref, residual_ref_buf, weight, EPS)
        out_ref, residual_ref = input_ref, residual_ref_buf

    # bf16 carries an 8-bit mantissa, so one ulp is a 2^-8 ~= 7.8e-3 relative step
    # and rtol=1e-2 only expresses 1.28 ulp. The fp32 reference rounds in a
    # different order than the kernel, and a sweep this wide reliably lands on a
    # 1-2 ulp disagreement (measured worst case 1.75 ulp over the 5120/8192 hidden
    # sizes). 1.5e-2 is the tightest bound that clears that noise: it still catches
    # a systematic 0.75% deviation, whereas 2e-2 would let 1% through. The
    # flashinfer path shares the kernel's rounding order, so it keeps 1e-2.
    out_rtol = 1.5e-2 if cast_x_before_out_mul else 1e-2
    torch.testing.assert_close(input_sglang, out_ref, atol=1e-2, rtol=out_rtol)
    torch.testing.assert_close(residual_sglang, residual_ref, atol=1e-2, rtol=1e-2)


class TestFusedScaledAddRMSNorm(CustomTestCase):
    def test_rejects_overlapping_buffers(self) -> None:
        from sglang.kernels.ops.layernorm.norm import fused_scaled_add_rmsnorm

        input = torch.randn(1, 512, device=DEVICE, dtype=DTYPE)
        residual = input.view_as(input)
        weight = torch.randn(512, device=DEVICE, dtype=DTYPE)

        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            fused_scaled_add_rmsnorm(
                input, residual, weight, SCALED_EPS, input_scale=0.22
            )

    def test_correctness(self) -> None:
        from sglang.kernels.ops.layernorm.norm import fused_scaled_add_rmsnorm

        input_scale = 0.22
        for dtype, batch_size, hidden_size in itertools.product(
            (torch.float16, torch.bfloat16),
            (1, 9, 256),
            (512, 1536, 8192),
        ):
            with self.subTest(
                dtype=dtype, batch_size=batch_size, hidden_size=hidden_size
            ):
                torch.manual_seed(0)
                input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
                residual = torch.randn_like(input)
                weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
                expected_input, expected_residual = forward_scaled_reference(
                    input, residual, weight, SCALED_EPS, input_scale
                )

                fused_scaled_add_rmsnorm(
                    input, residual, weight, SCALED_EPS, input_scale
                )

                torch.testing.assert_close(input, expected_input, atol=1e-2, rtol=1e-2)
                torch.testing.assert_close(residual, expected_residual, atol=0, rtol=0)

    def test_torch_compile_fullgraph(self) -> None:
        from sglang.kernels.ops.layernorm.norm import fused_scaled_add_rmsnorm

        captured_graphs = []

        def backend(graph_module, _example_inputs):
            captured_graphs.append(graph_module)
            return graph_module.forward

        def fn(input, residual, weight):
            fused_scaled_add_rmsnorm(input, residual, weight, SCALED_EPS, 0.22)
            return input, residual

        torch.manual_seed(0)
        input = torch.randn(1, 1536, device=DEVICE, dtype=DTYPE)
        residual = torch.randn_like(input)
        weight = torch.randn(1536, device=DEVICE, dtype=DTYPE)
        expected_input, expected_residual = forward_scaled_reference(
            input, residual, weight, SCALED_EPS, 0.22
        )

        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        actual_input, actual_residual = compiled(input, residual, weight)

        torch.testing.assert_close(actual_input, expected_input, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)
        self.assertEqual(len(captured_graphs), 1)
        self.assertTrue(
            any(
                node.target == torch.ops.sglang.fused_scaled_add_rmsnorm
                for node in captured_graphs[0].graph.nodes
            )
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
