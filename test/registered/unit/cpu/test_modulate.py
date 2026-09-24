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

DTYPE_PAIRS = [
    (torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, torch.float32),
    (torch.float16, torch.float16),
    (torch.float16, torch.float32),
]


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

    @pytest.mark.parametrize("input_dtype,param_dtype", DTYPE_PAIRS)
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

    @pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(
        "gate_type,norm_dtype,param_type,norm_type",
        [
            ("input", None, "input", "rms"),
            ("fp32", torch.float32, "input", "layer"),
            (None, None, "fp32", "layer"),
        ],
    )
    def test_fused_scale_residual_norm_scale_shift(
        self,
        input_dtype,
        gate_type,
        norm_dtype,
        param_type,
        norm_type,
    ):
        B, S, D = 2, 4, 67

        x = torch.randn(B, S, D, dtype=input_dtype)
        residual = torch.randn(B, S, D, dtype=input_dtype)

        gate_dtype = (
            input_dtype
            if gate_type == "input"
            else torch.float32
            if gate_type == "fp32"
            else None
        )
        param_dtype = input_dtype if param_type == "input" else torch.float32

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


TIMESTEP_BATCHES = [1, 8, 128]
TIMESTEP_DIMS = [31, 32, 128, 257, 512]
TIMESTEP_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]


def timestep_embedding_reference(
    timesteps,
    dim,
    *,
    flip_sin_to_cos=False,
    downscale_freq_shift=1,
    scale=1,
    max_period=10000,
):

    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    timesteps = timesteps.to(torch.float32)

    half_dim = dim // 2

    exponent = -torch.log(
        torch.tensor(max_period, dtype=torch.float32, device=timesteps.device)
    ) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )

    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)

    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb

    emb = torch.cat(
        [
            torch.sin(emb),
            torch.cos(emb),
        ],
        dim=-1,
    )

    if flip_sin_to_cos:
        emb = torch.cat(
            [
                emb[:, half_dim:],
                emb[:, :half_dim],
            ],
            dim=-1,
        )

    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


@pytest.mark.parametrize("batch_size", TIMESTEP_BATCHES)
@pytest.mark.parametrize("dim", TIMESTEP_DIMS)
@pytest.mark.parametrize("dtype", TIMESTEP_DTYPES)
@pytest.mark.parametrize(
    "flip_sin_to_cos,downscale_freq_shift,scale",
    [
        (True, 0, 1),
        (False, 1, 1),
        (True, 1, 0.01),
    ],
)
def test_timestep_embedding_cpu_matches_diffusers(
    batch_size,
    dim,
    dtype,
    flip_sin_to_cos,
    downscale_freq_shift,
    scale,
):
    timesteps = torch.randint(low=0, high=1000, size=(batch_size,), device="cpu").to(
        dtype
    )

    kwargs = dict(
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
        scale=scale,
        max_period=10000,
    )

    actual = torch.ops.sgl_kernel.timestep_embedding_cpu(
        timesteps,
        dim,
        kwargs["flip_sin_to_cos"],
        kwargs["downscale_freq_shift"],
        kwargs["scale"],
        kwargs["max_period"],
    )

    expected = timestep_embedding_reference(timesteps, dim, **kwargs)

    assert actual.dtype == torch.float32
    assert actual.shape == expected.shape

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
