from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import sglang.multimodal_gen.runtime.models.upsampler.latent_upsampler as lu_mod
from sglang.kernels.ops.diffusion.group_norm_silu import apply_group_norm_silu
from sglang.multimodal_gen.runtime.models.upsampler.latent_upsampler import (
    LatentUpsampler,
    ResBlock,
    SpatialRationalResampler,
)


def _resblock_eager_reference(block: ResBlock, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = block.activation(block.norm1(block.conv1(x)))  # fused site
    x = block.norm2(block.conv2(x))
    return block.activation(x + residual)


def _latent_upsampler_eager_reference(
    upsampler: LatentUpsampler, latent: torch.Tensor
) -> torch.Tensor:
    from einops import rearrange

    b, _, f, _, _ = latent.shape
    if upsampler.dims == 2:
        x = rearrange(latent, "b c f h w -> (b f) c h w")
        x = upsampler.initial_activation(
            upsampler.initial_norm(upsampler.initial_conv(x))
        )
        for block in upsampler.res_blocks:
            x = _resblock_eager_reference(block, x)
        x = upsampler.upsampler(x)
        for block in upsampler.post_upsample_res_blocks:
            x = _resblock_eager_reference(block, x)
        x = upsampler.final_conv(x)
        return rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

    x = upsampler.initial_activation(
        upsampler.initial_norm(upsampler.initial_conv(latent))
    )
    for block in upsampler.res_blocks:
        x = _resblock_eager_reference(block, x)
    if upsampler.temporal_upsample:
        x = upsampler.upsampler(x)[:, :, 1:, :, :]
    elif isinstance(upsampler.upsampler, SpatialRationalResampler):
        x = upsampler.upsampler(x)
    else:
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = upsampler.upsampler(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
    for block in upsampler.post_upsample_res_blocks:
        x = _resblock_eager_reference(block, x)
    return upsampler.final_conv(x)


@pytest.mark.parametrize(
    "batch,channels,dims,spatial",
    [
        (1, 64, 2, (16, 16)),
        (2, 64, 2, (8, 24)),
        (1, 128, 3, (2, 8, 8)),
    ],
)
def test_resblock_forward_parity(batch, channels, dims, spatial):
    torch.manual_seed(0)
    block = ResBlock(channels=channels, dims=dims).eval()
    torch.manual_seed(1)
    x = torch.randn(batch, channels, *spatial, dtype=torch.float32)

    with torch.no_grad():
        out = block(x)
        ref = _resblock_eager_reference(block, x)

    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "dims,latent_shape,mid_channels,num_blocks_per_stage,rational_resampler",
    [
        (2, (1, 32, 2, 16, 16), 64, 2, False),
        (3, (1, 32, 2, 16, 16), 64, 2, False),
        (3, (1, 32, 2, 16, 16), 64, 2, True),
    ],
)
def test_latent_upsampler_forward_parity(
    dims, latent_shape, mid_channels, num_blocks_per_stage, rational_resampler
):
    torch.manual_seed(2)
    upsampler = LatentUpsampler(
        in_channels=latent_shape[1],
        mid_channels=mid_channels,
        num_blocks_per_stage=num_blocks_per_stage,
        dims=dims,
        spatial_upsample=True,
        temporal_upsample=False,
        spatial_scale=2.0,
        rational_resampler=rational_resampler,
    ).eval()
    torch.manual_seed(3)
    latent = torch.randn(*latent_shape, dtype=torch.float32)

    with torch.no_grad():
        out = upsampler(latent)
        ref = _latent_upsampler_eager_reference(upsampler, latent)

    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


def test_resblock_fuses_exactly_one_site():
    torch.manual_seed(4)
    block = ResBlock(channels=64, dims=2).eval()
    x = torch.randn(1, 64, 16, 16, dtype=torch.float32)

    with patch.object(
        lu_mod, "apply_group_norm_silu", wraps=lu_mod.apply_group_norm_silu
    ) as spy:
        with torch.no_grad():
            block(x)
    assert spy.call_count == 1


@pytest.mark.parametrize(
    "dims,num_blocks_per_stage,rational_resampler,expected_calls",
    [
        (2, 2, False, 1 + 2 * 2),
        (2, 4, False, 1 + 4 * 2),
        (3, 2, False, 1 + 2 * 2),
        (3, 2, True, 1 + 2 * 2),
    ],
)
def test_latent_upsampler_fuses_expected_sites(
    dims, num_blocks_per_stage, rational_resampler, expected_calls
):
    torch.manual_seed(5)
    upsampler = LatentUpsampler(
        in_channels=32,
        mid_channels=64,
        num_blocks_per_stage=num_blocks_per_stage,
        dims=dims,
        spatial_upsample=True,
        temporal_upsample=False,
        spatial_scale=2.0,
        rational_resampler=rational_resampler,
    ).eval()
    latent = torch.randn(1, 32, 2, 16, 16, dtype=torch.float32)

    with patch.object(
        lu_mod, "apply_group_norm_silu", wraps=lu_mod.apply_group_norm_silu
    ) as spy:
        with torch.no_grad():
            upsampler(latent)
    assert spy.call_count == expected_calls


# CUDA Triton fast path -------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton fused group_norm_silu requires CUDA",
)

# bf16 keeps kernel-level tolerance because its fp32-equivalent exponent range
# absorbs multi-layer conv drift; fp16 needs a looser tolerance on the e2e
# upsampler test where 8+ downstream convs amplify fused-vs-eager rounding.
_RESBLOCK_TOL = {torch.bfloat16: (7e-2, 2e-2), torch.float16: (3e-3, 3e-3)}
_UPSAMPLER_TOL = {torch.bfloat16: (7e-2, 2e-2), torch.float16: (2e-2, 1e-1)}


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,channels,dims,spatial",
    [
        (1, 64, 2, (16, 16)),
        (1, 128, 3, (2, 8, 8)),
    ],
)
def test_resblock_forward_parity_cuda(dtype, batch, channels, dims, spatial):
    torch.manual_seed(0)
    device = torch.device("cuda")
    block = ResBlock(channels=channels, dims=dims).to(device=device, dtype=dtype).eval()
    torch.manual_seed(1)
    x = torch.randn(batch, channels, *spatial, device=device, dtype=dtype)

    with torch.no_grad():
        out = block(x)
        ref = _resblock_eager_reference(block, x)

    atol, rtol = _RESBLOCK_TOL[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dims,latent_shape,mid_channels,num_blocks_per_stage,rational_resampler",
    [
        (2, (1, 32, 2, 16, 16), 64, 2, False),
        (3, (1, 32, 2, 16, 16), 64, 2, False),
        (3, (1, 32, 2, 16, 16), 64, 2, True),
    ],
)
def test_latent_upsampler_forward_parity_cuda(
    dtype, dims, latent_shape, mid_channels, num_blocks_per_stage, rational_resampler
):
    torch.manual_seed(2)
    device = torch.device("cuda")
    upsampler = (
        LatentUpsampler(
            in_channels=latent_shape[1],
            mid_channels=mid_channels,
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
    latent = torch.randn(*latent_shape, device=device, dtype=dtype)

    with torch.no_grad():
        out = upsampler(latent)
        ref = _latent_upsampler_eager_reference(upsampler, latent)

    atol, rtol = _UPSAMPLER_TOL[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@requires_cuda
def test_resblock_actually_uses_triton_kernel_cuda():
    from sglang.kernels.ops.diffusion.triton import group_norm_silu as triton_mod

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
