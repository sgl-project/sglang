from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import sglang.multimodal_gen.runtime.models.upsampler.latent_upsampler as lu_mod
from sglang.kernels.ops.diffusion import apply_group_norm_silu
from sglang.multimodal_gen.runtime.models.upsampler.latent_upsampler import (
    LatentUpsampler,
    ResBlock,
)


def _resblock_eager_reference(block: ResBlock, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = block.activation(block.norm1(block.conv1(x)))  # fused site
    x = block.norm2(block.conv2(x))
    return block.activation(x + residual)


def _latent_upsampler_eager_reference(upsampler, latent):
    with patch.object(
        lu_mod, "apply_group_norm_silu", side_effect=lambda x, norm, act: act(norm(x))
    ):
        return upsampler(latent)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _parity_cases(common, cpu_only):
    return [
        pytest.param(
            device, dtype, *case, marks=requires_cuda if device == "cuda" else ()
        )
        for device, dtype in [
            ("cpu", torch.float32),
            ("cuda", torch.bfloat16),
            ("cuda", torch.float16),
        ]
        for case in common + (cpu_only if device == "cpu" else [])
    ]


# Downstream convolutions amplify fp16 drift in the complete upsampler.
_RESBLOCK_TOL = {
    torch.float32: (0, 0),
    torch.bfloat16: (7e-2, 2e-2),
    torch.float16: (3e-3, 3e-3),
}
_UPSAMPLER_TOL = {
    torch.float32: (0, 0),
    torch.bfloat16: (7e-2, 2e-2),
    torch.float16: (2e-2, 1e-1),
}


@pytest.mark.parametrize(
    "device,dtype,batch,channels,dims,spatial",
    _parity_cases(
        [(1, 64, 2, (16, 16)), (1, 128, 3, (2, 8, 8))],
        cpu_only=[(2, 64, 2, (8, 24))],
    ),
)
def test_resblock_forward_parity(device, dtype, batch, channels, dims, spatial):
    torch.manual_seed(0)
    block = ResBlock(channels=channels, dims=dims).to(device=device, dtype=dtype).eval()
    torch.manual_seed(1)
    x = torch.randn(batch, channels, *spatial, device=device, dtype=dtype)

    with (
        torch.no_grad(),
        patch.object(
            lu_mod, "apply_group_norm_silu", wraps=lu_mod.apply_group_norm_silu
        ) as fused,
    ):
        out = block(x)
    assert fused.call_count == 1
    with torch.no_grad():
        ref = _resblock_eager_reference(block, x)
    atol, rtol = _RESBLOCK_TOL[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "device,dtype,dims,num_blocks_per_stage,rational_resampler",
    _parity_cases(
        [(2, 2, False), (3, 2, False), (3, 2, True)],
        cpu_only=[(2, 4, False)],
    ),
)
def test_latent_upsampler_forward_parity(
    device, dtype, dims, num_blocks_per_stage, rational_resampler
):
    torch.manual_seed(2)
    upsampler = (
        LatentUpsampler(
            in_channels=32,
            mid_channels=64,
            num_blocks_per_stage=num_blocks_per_stage,
            dims=dims,
            spatial_upsample=True,
            temporal_upsample=False,
            spatial_scale=2.0,
            rational_resampler=rational_resampler,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    torch.manual_seed(3)
    latent = torch.randn(1, 32, 2, 16, 16, device=device, dtype=dtype)

    with (
        torch.no_grad(),
        patch.object(
            lu_mod, "apply_group_norm_silu", wraps=lu_mod.apply_group_norm_silu
        ) as fused,
    ):
        out = upsampler(latent)
    assert fused.call_count == 1 + 2 * num_blocks_per_stage
    with torch.no_grad():
        ref = _latent_upsampler_eager_reference(upsampler, latent)
    atol, rtol = _UPSAMPLER_TOL[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@requires_cuda
def test_resblock_actually_uses_triton_kernel_cuda():
    from sglang.kernels.ops.diffusion.norm import group_norm_silu_triton as triton_mod

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block = ResBlock(channels=64, dims=2).to(device=device, dtype=dtype).eval()
    x = torch.randn(1, 64, 16, 16, device=device, dtype=dtype)

    with patch.object(
        triton_mod,
        "triton_group_norm_silu",
        wraps=triton_mod.triton_group_norm_silu,
    ) as spy:
        with torch.no_grad():
            block(x)
    assert spy.call_count >= 1


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dims,latent_shape,mid_channels",
    [
        (2, (1, 32, 2, 16, 16), 64),
        (3, (1, 32, 2, 16, 16), 64),
    ],
)
def test_initial_groupnorm_silu_parity_cuda_local(
    dtype, dims, latent_shape, mid_channels
):
    # Sharp parity at the fused initial-norm boundary, before downstream convs
    # can amplify drift; uses kernel-level tolerance instead of the looser e2e.
    torch.manual_seed(2)
    device = torch.device("cuda")
    upsampler = (
        LatentUpsampler(
            in_channels=latent_shape[1],
            mid_channels=mid_channels,
            num_blocks_per_stage=2,
            dims=dims,
            spatial_upsample=True,
            temporal_upsample=False,
            spatial_scale=2.0,
            rational_resampler=False,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    torch.manual_seed(3)
    latent = torch.randn(*latent_shape, device=device, dtype=dtype)

    with torch.no_grad():
        if dims == 2:
            from einops import rearrange

            b, _, f, _, _ = latent.shape
            x_in = rearrange(latent, "b c f h w -> (b f) c h w")
        else:
            x_in = latent
        x_after_conv = upsampler.initial_conv(x_in)
        out_fused = apply_group_norm_silu(
            x_after_conv, upsampler.initial_norm, upsampler.initial_activation
        )
        out_eager = upsampler.initial_activation(upsampler.initial_norm(x_after_conv))

    atol, rtol = _RESBLOCK_TOL[dtype]
    torch.testing.assert_close(out_fused, out_eager, atol=atol, rtol=rtol)
