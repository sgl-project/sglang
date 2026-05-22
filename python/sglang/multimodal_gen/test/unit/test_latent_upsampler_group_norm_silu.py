"""CPU parity tests for the LTX-2 latent upsampler GroupNorm+SiLU fusion.

The `LatentUpsampler` (and its `ResBlock` building block) are run as the
``spatial_upsampler`` component in every LTX-2 two-stage pipeline call --
auto-resolved in `runtime/pipelines/ltx_2_pipeline.py` from one of several
candidate checkpoint paths. Unlike the LTX-2 *audio* VAE (whose default
`norm_type='pixel'` config skips the fused kernel), the upsampler uses
``torch.nn.GroupNorm`` unconditionally, so this wiring fires on the
production path.

These tests pin three properties:

1. **Output parity** -- the wired forward is numerically identical to an
   explicit eager reference that does `activation(norm(x))` directly.
   Tolerance is zero on CPU because ``apply_group_norm_silu``'s CPU fallback
   IS ``activation(norm(x))``.
2. **Helper call counts** -- the wired forward calls
   ``apply_group_norm_silu`` exactly the number of sites we intended to
   fuse (locks the wiring against accidental removal or duplication).
3. **Residual-add site stays eager** -- the second norm site inside
   ResBlock (``silu(norm2(x) + residual)``) is *not* routed through the
   current helper, since the helper doesn't cover the norm+add+silu
   pattern. This guards against accidentally rerouting it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import sglang.multimodal_gen.runtime.models.upsampler.latent_upsampler as lu_mod
from sglang.multimodal_gen.runtime.models.upsampler.latent_upsampler import (
    LatentUpsampler,
    ResBlock,
)


# -- Fixtures / helpers ----------------------------------------------------


def _seed(seed: int) -> None:
    torch.manual_seed(seed)


def _resblock_eager_reference(block: ResBlock, x: torch.Tensor) -> torch.Tensor:
    """Recompute ResBlock.forward using only ``activation(norm(x))`` for the
    fusable site -- matches the helper's CPU fallback exactly."""
    residual = x
    x = block.conv1(x)
    x = block.activation(block.norm1(x))  # fused site
    x = block.conv2(x)
    x = block.norm2(x)
    x = block.activation(x + residual)  # eager-only residual-add site
    return x


def _latent_upsampler_eager_reference(
    upsampler: LatentUpsampler, latent: torch.Tensor
) -> torch.Tensor:
    from einops import rearrange

    b, _, f, _, _ = latent.shape
    if upsampler.dims == 2:
        x = rearrange(latent, "b c f h w -> (b f) c h w")
        x = upsampler.initial_conv(x)
        x = upsampler.initial_activation(upsampler.initial_norm(x))  # fused site
        for block in upsampler.res_blocks:
            x = _resblock_eager_reference(block, x)
        x = upsampler.upsampler(x)
        for block in upsampler.post_upsample_res_blocks:
            x = _resblock_eager_reference(block, x)
        x = upsampler.final_conv(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        return x
    else:
        x = upsampler.initial_conv(latent)
        x = upsampler.initial_activation(upsampler.initial_norm(x))  # fused site
        for block in upsampler.res_blocks:
            x = _resblock_eager_reference(block, x)
        if upsampler.temporal_upsample:
            x = upsampler.upsampler(x)
            x = x[:, :, 1:, :, :]
        else:
            x = rearrange(x, "b c f h w -> (b f) c h w")
            x = upsampler.upsampler(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        for block in upsampler.post_upsample_res_blocks:
            x = _resblock_eager_reference(block, x)
        x = upsampler.final_conv(x)
        return x


# -- ResBlock parity -------------------------------------------------------


@pytest.mark.parametrize(
    "batch,channels,dims,spatial",
    [
        (1, 64, 2, (16, 16)),
        (2, 64, 2, (8, 24)),
        (1, 128, 3, (2, 8, 8)),
    ],
)
def test_resblock_forward_parity(batch, channels, dims, spatial):
    """ResBlock.forward (with one fused site) matches explicit eager."""
    _seed(0)
    block = ResBlock(channels=channels, dims=dims).eval()

    _seed(1)
    x = torch.randn(batch, channels, *spatial, dtype=torch.float32)

    with torch.no_grad():
        out_wired = block(x)
        out_ref = _resblock_eager_reference(block, x)

    torch.testing.assert_close(out_wired, out_ref, atol=0.0, rtol=0.0)


# -- LatentUpsampler parity (2D + 3D) --------------------------------------


@pytest.mark.parametrize(
    "dims,latent_shape,mid_channels,num_blocks_per_stage",
    [
        # 2D conv path
        (2, (1, 32, 2, 16, 16), 64, 2),
        # 3D conv path with spatial upsample
        (3, (1, 32, 2, 16, 16), 64, 2),
    ],
)
def test_latent_upsampler_forward_parity(
    dims, latent_shape, mid_channels, num_blocks_per_stage
):
    """LatentUpsampler.forward matches explicit eager across 2D / 3D paths."""
    _seed(2)
    upsampler = LatentUpsampler(
        in_channels=latent_shape[1],
        mid_channels=mid_channels,
        num_blocks_per_stage=num_blocks_per_stage,
        dims=dims,
        spatial_upsample=True,
        temporal_upsample=False,
        spatial_scale=2.0,
        rational_resampler=False,
    ).eval()

    _seed(3)
    latent = torch.randn(*latent_shape, dtype=torch.float32)

    with torch.no_grad():
        out_wired = upsampler(latent)
        out_ref = _latent_upsampler_eager_reference(upsampler, latent)

    torch.testing.assert_close(out_wired, out_ref, atol=0.0, rtol=0.0)


# -- Helper call counts ----------------------------------------------------


def test_resblock_fuses_exactly_one_site():
    """Each ResBlock.forward must call apply_group_norm_silu exactly once;
    the residual-add second-norm site stays on the eager path until a
    norm+add+silu helper exists."""
    _seed(4)
    block = ResBlock(channels=64, dims=2).eval()
    x = torch.randn(1, 64, 16, 16, dtype=torch.float32)

    real = lu_mod.apply_group_norm_silu
    with patch.object(lu_mod, "apply_group_norm_silu", wraps=real) as mock_helper:
        with torch.no_grad():
            _ = block(x)
    assert mock_helper.call_count == 1, (
        f"ResBlock should fuse exactly 1 site; got {mock_helper.call_count}"
    )


@pytest.mark.parametrize(
    "dims,num_blocks_per_stage,expected_calls",
    [
        # 1 initial site + (num_blocks_per_stage * 2 ResBlocks, 1 each)
        (2, 2, 1 + 2 * 2),
        (2, 4, 1 + 4 * 2),
        (3, 2, 1 + 2 * 2),
    ],
)
def test_latent_upsampler_fuses_expected_sites(
    dims, num_blocks_per_stage, expected_calls
):
    """LatentUpsampler.forward fuses 1 initial site + 1 per ResBlock (pre
    + post upsample). The 2nd norm in each ResBlock stays eager."""
    _seed(5)
    upsampler = LatentUpsampler(
        in_channels=32,
        mid_channels=64,
        num_blocks_per_stage=num_blocks_per_stage,
        dims=dims,
        spatial_upsample=True,
        temporal_upsample=False,
        spatial_scale=2.0,
        rational_resampler=False,
    ).eval()
    latent = torch.randn(1, 32, 2, 16, 16, dtype=torch.float32)

    real = lu_mod.apply_group_norm_silu
    with patch.object(lu_mod, "apply_group_norm_silu", wraps=real) as mock_helper:
        with torch.no_grad():
            _ = upsampler(latent)
    assert mock_helper.call_count == expected_calls, (
        f"LatentUpsampler should fuse {expected_calls} sites; "
        f"got {mock_helper.call_count}"
    )


# -- Module wiring smoke test ---------------------------------------------


def test_latent_upsampler_module_wires_fused_helper():
    """latent_upsampler.py imports apply_group_norm_silu at module scope."""
    assert hasattr(lu_mod, "apply_group_norm_silu")


# -- CUDA production fast path (Triton fused kernel) ----------------------
#
# The CPU/fp32 tests above prove the wiring is structurally correct and the
# helper's eager fallback matches an explicit reference exactly. They do
# *not* exercise the Triton fused path, because ``apply_group_norm_silu``
# gates the fused kernel on ``x.is_cuda`` and bf16/fp16 dtypes.
#
# The tests below run the same ``ResBlock.forward`` / ``LatentUpsampler.forward``
# wired modules on CUDA with bf16 and fp16, so the helper actually invokes
# ``triton_group_norm_silu``. Parity is against an explicit eager reference
# that uses the same module instance but bypasses the helper, so any
# numerical gap is purely fused-vs-eager. Tolerances match the kernel-side
# test (`jit_kernel/tests/diffusion/test_group_norm_silu.py::_tol`).

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton fused group_norm_silu requires CUDA",
)

# Tolerances. ResBlock isolates exactly one fused site (norm1 + activation)
# from the output by 1 conv layer, so the kernel-side tolerance suffices.
# LatentUpsampler wraps the fused site(s) behind 8+ conv2d/conv3d layers
# (initial_conv -> 4 pre-upsample ResBlocks -> upsampler -> 4 post-upsample
# ResBlocks -> final_conv with default num_blocks_per_stage=4), and each
# downstream conv amplifies fp16 quantization differences from the
# fused-vs-eager path. bf16 has fp32-equivalent exponent range so multi-layer
# amplification stays tight; fp16 needs an intentionally looser tolerance.
_BF16_TOL_KERNEL = (7e-2, 2e-2)
_FP16_TOL_KERNEL = (3e-3, 3e-3)
_BF16_TOL_MULTI_LAYER = (7e-2, 2e-2)
_FP16_TOL_MULTI_LAYER = (2e-2, 1e-1)

_RESBLOCK_TOL = {
    torch.bfloat16: _BF16_TOL_KERNEL,
    torch.float16: _FP16_TOL_KERNEL,
}
_UPSAMPLER_TOL = {
    torch.bfloat16: _BF16_TOL_MULTI_LAYER,
    torch.float16: _FP16_TOL_MULTI_LAYER,
}


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
    """ResBlock.forward on CUDA bf16/fp16 fires the Triton kernel via the
    helper and matches the eager reference within bf16/fp16 tolerance."""
    _seed(0)
    device = torch.device("cuda")
    block = ResBlock(channels=channels, dims=dims).to(device=device, dtype=dtype).eval()

    _seed(1)
    x = torch.randn(batch, channels, *spatial, device=device, dtype=dtype)

    with torch.no_grad():
        out_wired = block(x)
        out_ref = _resblock_eager_reference(block, x)

    atol, rtol = _RESBLOCK_TOL[dtype]
    torch.testing.assert_close(out_wired, out_ref, atol=atol, rtol=rtol)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dims,latent_shape,mid_channels,num_blocks_per_stage",
    [
        # 2D conv path
        (2, (1, 32, 2, 16, 16), 64, 2),
        # 3D conv path with spatial upsample (the production LTX-2 path)
        (3, (1, 32, 2, 16, 16), 64, 2),
    ],
)
def test_latent_upsampler_forward_parity_cuda(
    dtype, dims, latent_shape, mid_channels, num_blocks_per_stage
):
    """LatentUpsampler.forward on CUDA bf16/fp16 end-to-end parity. Each
    forward fires the Triton kernel 1 + 2 * num_blocks_per_stage times
    (initial + pre/post-upsample ResBlocks). The fp16 tolerance is looser
    than ResBlock's because 8+ conv layers downstream amplify fp16
    quantization (see _UPSAMPLER_TOL comment above)."""
    _seed(2)
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
            rational_resampler=False,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )

    _seed(3)
    latent = torch.randn(*latent_shape, device=device, dtype=dtype)

    with torch.no_grad():
        out_wired = upsampler(latent)
        out_ref = _latent_upsampler_eager_reference(upsampler, latent)

    atol, rtol = _UPSAMPLER_TOL[dtype]
    torch.testing.assert_close(out_wired, out_ref, atol=atol, rtol=rtol)


@requires_cuda
def test_resblock_actually_uses_triton_kernel_cuda():
    """Verify the helper truly routes to the Triton path on CUDA bf16 (not
    just falls back to eager). Asserts ``triton_group_norm_silu`` is invoked
    by patching the import the helper does lazily."""
    from unittest.mock import patch

    from sglang.jit_kernel.diffusion.triton import group_norm_silu as triton_mod

    _seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block = ResBlock(channels=64, dims=2).to(device=device, dtype=dtype).eval()
    x = torch.randn(1, 64, 16, 16, device=device, dtype=dtype)

    real = triton_mod.triton_group_norm_silu
    with patch.object(
        triton_mod, "triton_group_norm_silu", wraps=real
    ) as mock_triton:
        with torch.no_grad():
            _ = block(x)
    assert mock_triton.call_count >= 1, (
        "Expected the Triton fused kernel to fire at least once on CUDA bf16; "
        f"got call_count={mock_triton.call_count} (helper fell back to eager?)"
    )
