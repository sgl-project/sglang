import sys

import pytest
import sgl_kernel  # noqa: F401
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import precision

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")
register_cpu_ci(est_time=10, suite="base-b-test-cpu-arm64")

torch.manual_seed(1234)

eps = 1e-6


class TestDiffusionNorm:
    def rmsnorm_ref(
        self,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        x_fp32 = x.float()
        variance = x_fp32.square().mean(dim=-1, keepdim=True)
        out = x_fp32 * torch.rsqrt(variance + eps)

        if weight is not None:
            out = out * weight.float()

        return out

    @pytest.mark.parametrize(
        "input_dtype,param_dtype",
        [
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.float16),
            (torch.float16, torch.float32),
        ],
    )
    @pytest.mark.parametrize("broadcast_c", [False, True])
    def test_fused_scale_shift(
        self,
        input_dtype,
        param_dtype,
        broadcast_c,
    ):
        B, S, D = 2, 4, 67
        x = torch.randn(B, S, D, dtype=input_dtype)

        if broadcast_c:
            # hidden dimension broadcast -> stride_c == 0
            scale = torch.randn(B, 1, 1, dtype=param_dtype)
            shift = torch.randn(B, S, 1, dtype=param_dtype)
        else:
            # normal vector load -> stride_c == 1
            scale = torch.randn(B, 1, D, dtype=param_dtype)
            shift = torch.randn(B, S, D, dtype=param_dtype)

        scale_expanded = scale.expand_as(x)
        shift_expanded = shift.expand_as(x)

        if broadcast_c:
            assert scale_expanded.stride(2) == 0
            assert shift_expanded.stride(2) == 0
        else:
            assert scale_expanded.stride(2) == 1
            assert shift_expanded.stride(2) == 1

        out = torch.ops.sgl_kernel.fused_scale_shift_cpu(
            x,
            scale_expanded,
            shift_expanded,
            1.0,
        )

        ref = (x.float() * (1.0 + scale.float()) + shift.float()).to(input_dtype)

        torch.testing.assert_close(
            out,
            ref,
            atol=precision[input_dtype],
            rtol=precision[input_dtype],
        )

    @pytest.mark.parametrize(
        "input_dtype,param_dtype",
        [
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.float16),
            (torch.float16, torch.float32),
        ],
    )
    @pytest.mark.parametrize("norm_type", ["rms", "layer"])
    def test_fused_norm_scale_shift(
        self,
        input_dtype,
        param_dtype,
        norm_type,
    ):
        B, S, D = 2, 4, 67

        x = torch.randn(B, S, D, dtype=input_dtype)
        weight = torch.randn(D, dtype=torch.float32)
        bias = torch.randn(D, dtype=torch.float32) if norm_type == "layer" else None

        scale = torch.randn(B, 1, D, dtype=param_dtype)
        shift = torch.randn(B, S, D, dtype=param_dtype)

        scale_expanded = scale.expand_as(x)
        shift_expanded = shift.expand_as(x)

        out = torch.ops.sgl_kernel.fused_norm_scale_shift_cpu(
            x,
            weight,
            bias,
            scale_expanded,
            shift_expanded,
            norm_type,
            eps=eps,
        )

        if norm_type == "rms":
            normalized = self.rmsnorm_ref(x, weight, eps)
        else:
            normalized = torch.nn.functional.layer_norm(
                x.float(),
                (D,),
                weight,
                bias,
                eps,
            )

        # Match CUDA/CuTe activation-dtype boundary.
        normalized = normalized.to(input_dtype).float()

        ref = (normalized * (1.0 + scale.float()) + shift.float()).to(input_dtype)

        torch.testing.assert_close(
            out,
            ref,
            atol=precision[input_dtype],
            rtol=precision[input_dtype],
        )

    @pytest.mark.parametrize(
        "input_dtype,gate_dtype,norm_dtype,param_dtype,norm_type",
        [
            (
                torch.bfloat16,
                torch.bfloat16,
                None,
                torch.bfloat16,
                "rms",
            ),
            (
                torch.float16,
                torch.float16,
                None,
                torch.float16,
                "rms",
            ),
            (
                torch.bfloat16,
                torch.float32,
                torch.float32,
                torch.bfloat16,
                "layer",
            ),
            (
                torch.float16,
                torch.float32,
                torch.float32,
                torch.float16,
                "layer",
            ),
            (
                torch.bfloat16,
                None,
                None,
                torch.float32,
                "layer",
            ),
            (
                torch.float16,
                None,
                None,
                torch.float32,
                "layer",
            ),
        ],
    )
    def test_fused_scale_residual_norm_scale_shift(
        self,
        input_dtype,
        gate_dtype,
        norm_dtype,
        param_dtype,
        norm_type,
    ):
        B, S, D = 2, 4, 67

        x = torch.randn(B, S, D, dtype=input_dtype)
        residual = torch.randn(B, S, D, dtype=input_dtype)

        gate = torch.randn(D, dtype=gate_dtype) if gate_dtype is not None else None

        weight = torch.randn(D, dtype=norm_dtype) if norm_dtype is not None else None

        bias = (
            torch.randn(D, dtype=norm_dtype)
            if norm_dtype is not None and norm_type == "layer"
            else None
        )

        scale = torch.randn(B, 1, D, dtype=param_dtype)
        shift = torch.randn(B, S, D, dtype=param_dtype)

        scale_expanded = scale.expand_as(x)
        shift_expanded = shift.expand_as(x)

        gate_expanded = gate.view(1, 1, D).expand_as(x) if gate is not None else None

        out, residual_out = (
            torch.ops.sgl_kernel.fused_scale_residual_norm_scale_shift_cpu(
                residual,
                x,
                gate_expanded,
                weight,
                bias,
                scale_expanded,
                shift_expanded,
                norm_type,
                eps=eps,
            )
        )

        if gate is None:
            residual_fp32 = residual.float() + x.float()
        else:
            residual_fp32 = residual.float() + x.float() * gate.float()

        ref_residual = residual_fp32.to(input_dtype)
        norm_input = ref_residual.float()

        if norm_type == "rms":
            normalized = self.rmsnorm_ref(norm_input, weight, eps)
        else:
            normalized = torch.nn.functional.layer_norm(
                norm_input,
                (D,),
                weight.float() if weight is not None else None,
                bias.float() if bias is not None else None,
                eps,
            )

        # Match CUDA activation boundary after norm.
        normalized = normalized.to(input_dtype).float()

        ref_out = (normalized * (1.0 + scale.float()) + shift.float()).to(input_dtype)

        torch.testing.assert_close(
            residual_out,
            ref_residual,
            atol=precision[input_dtype],
            rtol=precision[input_dtype],
        )

        torch.testing.assert_close(
            out, ref_out, atol=precision[input_dtype], rtol=precision[input_dtype]
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
