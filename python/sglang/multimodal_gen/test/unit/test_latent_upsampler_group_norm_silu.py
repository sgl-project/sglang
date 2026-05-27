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
from sglang.jit_kernel.diffusion.group_norm_silu import apply_group_norm_silu
from sglang.multimodal_gen.runtime.models.upsampler.latent_upsampler import (
    LatentUpsampler,
    ResBlock,
    SpatialRationalResampler,
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
        # Mirror the three-branch dispatch in production
        # ``LatentUpsampler.forward`` (3D path): temporal_upsample uses a
        # 5D upsampler + temporal slice; SpatialRationalResampler is invoked
        # directly on the 5D tensor (its own forward flattens internally);
        # otherwise we flatten temporal-into-batch around a 4D upsampler.
        if upsampler.temporal_upsample:
            x = upsampler.upsampler(x)
            x = x[:, :, 1:, :, :]
        elif isinstance(upsampler.upsampler, SpatialRationalResampler):
            x = upsampler.upsampler(x)
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
    "dims,latent_shape,mid_channels,num_blocks_per_stage,rational_resampler",
    [
        # 2D conv path (rational_resampler=True is invalid for dims=2: the
        # 2D forward flattens before calling self.upsampler, but
        # SpatialRationalResampler.forward expects 5D input, so production
        # only supports rational_resampler with dims=3).
        (2, (1, 32, 2, 16, 16), 64, 2, False),
        # 3D conv path with Conv3d + PixelShuffleND spatial upsampler
        (3, (1, 32, 2, 16, 16), 64, 2, False),
        # 3D conv path with SpatialRationalResampler -- production loader
        # exposes rational_resampler=True via LTX-2 spatial upscaler configs,
        # and this branch hits a different upsampler dispatch in the forward.
        (3, (1, 32, 2, 16, 16), 64, 2, True),
    ],
)
def test_latent_upsampler_forward_parity(
    dims, latent_shape, mid_channels, num_blocks_per_stage, rational_resampler
):
    """LatentUpsampler.forward matches explicit eager across 2D / 3D paths
    and across the rational vs non-rational spatial upsampler dispatch."""
    _seed(2)
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
    "dims,num_blocks_per_stage,rational_resampler,expected_calls",
    [
        # 1 initial site + (num_blocks_per_stage * 2 ResBlocks, 1 each).
        # Site count is invariant under rational_resampler -- the upsampler
        # itself has no GroupNorm sites -- but we explicitly verify that.
        (2, 2, False, 1 + 2 * 2),
        (2, 4, False, 1 + 4 * 2),
        (3, 2, False, 1 + 2 * 2),
        (3, 2, True, 1 + 2 * 2),
    ],
)
def test_latent_upsampler_fuses_expected_sites(
    dims, num_blocks_per_stage, rational_resampler, expected_calls
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
        rational_resampler=rational_resampler,
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
    "dims,latent_shape,mid_channels,num_blocks_per_stage,rational_resampler",
    [
        # 2D conv path
        (2, (1, 32, 2, 16, 16), 64, 2, False),
        # 3D conv path with Conv3d + PixelShuffleND spatial upsampler
        (3, (1, 32, 2, 16, 16), 64, 2, False),
        # 3D conv path with SpatialRationalResampler (different upsampler
        # dispatch; production LTX-2 spatial upscaler configs use this).
        (3, (1, 32, 2, 16, 16), 64, 2, True),
    ],
)
def test_latent_upsampler_forward_parity_cuda(
    dtype,
    dims,
    latent_shape,
    mid_channels,
    num_blocks_per_stage,
    rational_resampler,
):
    """LatentUpsampler.forward on CUDA bf16/fp16 end-to-end parity. Each
    forward fires the Triton kernel 1 + 2 * num_blocks_per_stage times
    (initial + pre/post-upsample ResBlocks). The fp16 tolerance is looser
    than ResBlock's because 8+ conv layers downstream amplify fp16
    quantization (see _UPSAMPLER_TOL comment above); the localized
    fused-site test below gives the precise per-kernel signal."""
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
            rational_resampler=rational_resampler,
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
    """Local CUDA parity at the *boundary* of the fused initial GroupNorm
    + SiLU site, before any downstream conv can amplify differences.

    The end-to-end ``test_latent_upsampler_forward_parity_cuda`` necessarily
    uses a looser tolerance on fp16 because 8+ conv layers downstream
    accumulate quantization drift between the fused and eager paths -- which
    is honest about the e2e numerics but loose enough that a real kernel
    regression could slip through. This test isolates the fused site so we
    get a precise per-kernel signal: build the upsampler, run only the prefix
    of the forward up to (and including) the fused site, and compare the
    fused vs eager outputs of ``apply_group_norm_silu`` at the same tolerance
    the kernel-level test in
    ``jit_kernel/tests/diffusion/test_group_norm_silu.py`` uses
    (``_RESBLOCK_TOL`` -- bf16: 7e-2/2e-2; fp16: 3e-3/3e-3).
    """
    _seed(2)
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

    _seed(3)
    latent = torch.randn(*latent_shape, device=device, dtype=dtype)

    # Mirror the prefix of LatentUpsampler.forward up to the fused site.
    with torch.no_grad():
        if dims == 2:
            b, _, f, _, _ = latent.shape
            from einops import rearrange

            x_in = rearrange(latent, "b c f h w -> (b f) c h w")
        else:
            x_in = latent
        x_after_conv = upsampler.initial_conv(x_in)

        out_fused = apply_group_norm_silu(
            x_after_conv, upsampler.initial_norm, upsampler.initial_activation
        )
        out_eager = upsampler.initial_activation(
            upsampler.initial_norm(x_after_conv)
        )

    atol, rtol = _RESBLOCK_TOL[dtype]
    torch.testing.assert_close(out_fused, out_eager, atol=atol, rtol=rtol)
