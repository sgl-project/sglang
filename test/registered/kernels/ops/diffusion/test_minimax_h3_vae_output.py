"""MiniMax-H3 VAE output kernels and decoder batching against the eager path.

``blend`` below re-implements ``AutoencoderKL.blend`` with plain aten ops,
``revert_reference`` the ``VAEProcessor.revert_tensor`` Normalize + clamp.
Every kernel result must be ``torch.equal`` to its reference (NaN-aware only
where the input carries NaN); a tolerance here would hide a real bug.
"""

import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    can_use_minimax_h3_vae_denorm_clamp,
    can_use_minimax_h3_vae_temporal_blend_write,
    minimax_h3_vae_denorm_clamp,
    minimax_h3_vae_temporal_blend_write,
)
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    klvae as h3_klvae,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    processor as h3_processor,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kernels.minimax_h3_vae import _h3_install_gate, _h3_tiny_vae

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=70, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
# backend-specific: preserve ROCm SDPA/autocast batching parity coverage.
register_amd_ci(est_time=2, suite="nightly-amd-kernel-1-gpu", nightly=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

requires_h3_kernels = pytest.mark.skipif(
    not is_cuda_sm_at_least((10, 0)),
    reason="MiniMax-H3 VAE output kernels are gated to SM100+",
)

IMAGENET_MEAN = (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225)
IMAGENET_STD = (1 / 0.229, 1 / 0.224, 1 / 0.225)


@pytest.fixture
def inference():
    torch.manual_seed(20260919)
    with torch.inference_mode():
        yield


def blend(a, b, extent, dim):
    # AutoencoderKL.blend: fp32 ramp weights, a * (1 - w) then += b * w.
    extent = min(a.shape[dim], b.shape[dim], extent)
    positions = torch.arange(extent, device=b.device, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = extent
    weight_a = (1 - positions / extent).view(shape)
    weight_b = (positions / extent).view(shape)
    blended = a.narrow(dim, a.shape[dim] - extent, extent) * weight_a
    blended.add_(b.narrow(dim, 0, extent) * weight_b)
    return torch.cat((blended, b.narrow(dim, extent, b.shape[dim] - extent)), dim=dim)


def revert_reference(value, mean, std):
    mean = value.new_tensor(mean).view(1, 3, 1, 1, 1)
    std = value.new_tensor(std).view(1, 3, 1, 1, 1)
    return (value - mean).div_(std).clamp_(0, 1)


def assert_bitwise(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@requires_h3_kernels
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize(
    "strided,part_frames,overlap_frames,extent,count",
    [
        (False, 5, 3, 0, 4),
        (False, 8, 7, 7, 8),  # Reciprocal-first ramp plus an unblended tail.
        (True, 8, 7, 15, 8),  # Same ramp after clamping, with strided tensors.
        (False, 12, 12, 12, 12),
        (True, 15, 15, 15, 15),
        (False, 5, 3, 2, 0),  # Zero write leaves the destination intact.
    ],
)
def test_temporal_blend_write_matches_eager(
    strided, part_frames, overlap_frames, extent, count
):
    width = 24 if strided else 12
    part = torch.randn(2, 3, part_frames, 8, width, device="cuda")
    overlap = torch.randn(2, 3, overlap_frames, 8, width, device="cuda")
    storage = torch.full((2, 3, count + 6, 8, width), -111.0, device="cuda")
    out = storage
    if strided:
        part, overlap, out = part[..., 1::2], overlap[..., 1::2], storage[..., 1::2]
    assert can_use_minimax_h3_vae_temporal_blend_write(
        part, overlap, blend_extent=extent, out=out, start=3, count=count
    )
    expected = out.clone()
    expected[:, :, 3 : 3 + count] = blend(overlap, part, extent, 2)[:, :, :count]
    minimax_h3_vae_temporal_blend_write(
        part, overlap, blend_extent=extent, out=out, start=3, count=count
    )
    assert_bitwise(out, expected)
    if strided:
        assert torch.all(storage[..., ::2] == -111).item()


@requires_h3_kernels
@pytest.mark.usefixtures("inference")
def test_temporal_blend_write_without_overlap_and_alias_guard():
    part = torch.randn(1, 3, 5, 7, 11, device="cuda")
    out = torch.full((1, 3, 8, 7, 11), -9.0, device="cuda")
    assert can_use_minimax_h3_vae_temporal_blend_write(
        part, None, blend_extent=4, out=out, start=2, count=5
    )
    minimax_h3_vae_temporal_blend_write(
        part, None, blend_extent=4, out=out, start=2, count=5
    )
    assert torch.equal(out[:, :, 2:7], part)
    assert (
        torch.all(out[:, :, :2] == -9).item() and torch.all(out[:, :, 7:] == -9).item()
    )
    # Shared storage is rejected by the predicate, and by the launcher whenever
    # the byte range of the written frames meets the byte range of the frames
    # it reads (a bounds check, so interleaved views are rejected as well).
    assert not can_use_minimax_h3_vae_temporal_blend_write(
        out[:, :, :5], None, blend_extent=0, out=out, start=0, count=5
    )
    with pytest.raises(RuntimeError, match="alias"):
        minimax_h3_vae_temporal_blend_write(
            out[:, :, :5], None, blend_extent=0, out=out, start=0, count=5
        )
    with pytest.raises(RuntimeError, match="alias"):
        minimax_h3_vae_temporal_blend_write(
            part, out[:, :, :3], blend_extent=2, out=out, start=0, count=5
        )
    # The check is per frame range, not per view: with the frame axis outermost
    # in memory (B = C = 1), disjoint frame windows of one buffer are allowed.
    single = torch.full((1, 1, 8, 7, 11), -9.0, device="cuda")
    single[:, :, :3] = torch.randn(1, 1, 3, 7, 11, device="cuda")
    minimax_h3_vae_temporal_blend_write(
        single[:, :, :3], None, blend_extent=0, out=single, start=5, count=3
    )
    assert torch.equal(single[:, :, 5:8], single[:, :, :3])
    with pytest.raises(RuntimeError, match="count"):
        minimax_h3_vae_temporal_blend_write(
            part, None, blend_extent=0, out=out, start=0, count=6
        )


@requires_h3_kernels
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("strided", [False, True])
def test_denorm_clamp_matches_eager(strided):
    value = torch.randn(2, 3, 5, 8, 24 if strided else 12, device="cuda")
    if strided:
        value = value[..., 1::2]
    value[0, 0, 0, 0, :3] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], device="cuda"
    )
    assert can_use_minimax_h3_vae_denorm_clamp(
        value, mean=IMAGENET_MEAN, std=IMAGENET_STD
    )
    expected = revert_reference(value, IMAGENET_MEAN, IMAGENET_STD)
    actual = minimax_h3_vae_denorm_clamp(value, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    assert actual.is_contiguous()
    assert_bitwise(actual, expected)
    assert torch.isnan(actual[0, 0, 0, 0, 0]).item()
    assert actual[0, 0, 0, 0, 1].item() == 1.0 and actual[0, 0, 0, 0, 2].item() == 0.0


@requires_h3_kernels
@pytest.mark.usefixtures("inference")
def test_denorm_clamp_in_place_rewrites_a_contiguous_input():
    value = torch.randn(2, 3, 5, 8, 12, device="cuda")
    expected = revert_reference(value, IMAGENET_MEAN, IMAGENET_STD)
    assert can_use_minimax_h3_vae_denorm_clamp(
        value, mean=IMAGENET_MEAN, std=IMAGENET_STD, out=value
    )
    actual = minimax_h3_vae_denorm_clamp(
        value, mean=IMAGENET_MEAN, std=IMAGENET_STD, out=value
    )
    assert actual is value
    assert_bitwise(actual, expected)
    # Only the input itself may be the output, and only when it is contiguous.
    strided = torch.randn(2, 3, 5, 8, 24, device="cuda")[..., ::2]
    assert not can_use_minimax_h3_vae_denorm_clamp(
        strided, mean=IMAGENET_MEAN, std=IMAGENET_STD, out=strided
    )
    with pytest.raises(RuntimeError):
        minimax_h3_vae_denorm_clamp(
            strided, mean=IMAGENET_MEAN, std=IMAGENET_STD, out=strided
        )
    with pytest.raises(ValueError):
        minimax_h3_vae_denorm_clamp(
            value, mean=IMAGENET_MEAN, std=IMAGENET_STD, out=torch.empty_like(value)
        )


@requires_h3_kernels
@pytest.mark.usefixtures("inference")
def test_denorm_clamp_rejects_unsupported_inputs():
    value = torch.randn(2, 4, 5, 8, 12, device="cuda")
    assert not can_use_minimax_h3_vae_denorm_clamp(
        value, mean=IMAGENET_MEAN, std=IMAGENET_STD
    )
    assert not can_use_minimax_h3_vae_denorm_clamp(
        value[:, :3].half(), mean=IMAGENET_MEAN, std=IMAGENET_STD
    )
    with pytest.raises(RuntimeError):
        minimax_h3_vae_denorm_clamp(value, mean=IMAGENET_MEAN, std=IMAGENET_STD)


@requires_h3_kernels
@torch.no_grad()
def test_minimax_h3_temporal_blend_write_kernel_is_bit_exact(monkeypatch):
    """``_write_frames`` blends the window onto the previous overlap straight
    into the destination frames, bitwise equal to blend + slice copy, with and
    without an overlap."""
    vae = _h3_tiny_vae().cuda()
    gate = BitExactFusionGate("test temporal blend", per_signature=True)
    monkeypatch.setattr(h3_klvae, "_TEMPORAL_BLEND_FUSION", gate)
    overlap = torch.randn(1, 3, 6, 16, 16, device="cuda")
    part = torch.randn(1, 3, 6, 16, 16, device="cuda")
    expected = torch.zeros(1, 3, 11, 16, 16, device="cuda")
    expected[:, :, 2:7] = vae._blended_frames(part, overlap)[:, :, :5]
    expected[:, :, 7:11] = part[:, :, :4]
    dec = torch.zeros_like(expected)
    with patch.object(
        h3_klvae,
        "minimax_h3_vae_temporal_blend_write",
        wraps=h3_klvae.minimax_h3_vae_temporal_blend_write,
    ) as fused:
        vae._write_frames(dec, part, overlap, start=2, count=5)
        vae._write_frames(dec, part, None, start=7, count=4)
    assert fused.call_count == 2 and not gate.disabled
    # The blend and the plain copy are verified as separate signatures.
    assert gate.is_verified(True) and gate.is_verified(False)
    assert torch.equal(dec, expected)


@torch.no_grad()
@pytest.mark.parametrize("capacity", [2, 4, 8])
def test_minimax_h3_batching_is_close_but_not_bit_identical(capacity):
    """Tile/window batching vs one-at-a-time forwards on the tiny VAE under fp16
    autocast on any CUDA GPU, same kernels on both sides (only the leading
    dimension changes).

    Bit-identity does NOT hold: cuBLAS/cuDNN pick algorithms by batch size,
    deterministically run to run, so the differences are algorithm choice, not
    races. The bound below is one fp16 ulp of the operands."""
    vae = _h3_tiny_vae().cuda()
    gate = _h3_install_gate(vae)
    z = torch.randn(2, 4, 11, 16, 16, device="cuda")
    decode_batches = []
    eager_decode = vae.decode

    def recording_decode(latent):
        decode_batches.append(int(latent.shape[0]))
        return eager_decode(latent)

    def run(tiles, windows):
        vae.decoder_tile_batch, vae.window_batch = tiles, windows
        decode_batches.clear()
        with (
            torch.autocast("cuda", dtype=torch.float16),
            use_vae_fast_path(vae, True),
            patch.object(vae, "decode", recording_decode),
        ):
            frames = vae.decode_temporal(z)
            pixels = vae.processor.revert_tensor(frames)
        return frames.clone(), pixels.clone(), list(decode_batches)

    ref_frames, ref_pixels, ref_batches = run(1, 1)
    assert not gate.enabled
    # Two windows x nine tiles x two samples: 18 forwards of one tile (2 rows).
    assert ref_batches == [2] * 18
    frames, pixels, batches = run(capacity, capacity)
    # Both windows share a forward (4 rows per tile), `capacity` tiles at a time.
    assert max(batches) == min(capacity, 9) * 4 and sum(batches) == 36
    assert frames.dtype == torch.float32
    torch.testing.assert_close(frames, ref_frames, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(pixels, ref_pixels, atol=1e-3, rtol=1e-3)


@requires_h3_kernels
@torch.no_grad()
@pytest.mark.parametrize("ndim", [5, 4])
@pytest.mark.parametrize("runtime_owned", [False, True])
def test_minimax_h3_revert_tensor_kernel_is_bit_exact(monkeypatch, ndim, runtime_owned):
    """``VAEProcessor.revert_tensor`` runs the denorm kernel once per call,
    in place exactly when the caller owns the frames, and is bitwise equal to
    the eager Normalize + clamp chain for a clip and for one 4-D frame."""
    vae = _h3_tiny_vae().cuda()
    gate = BitExactFusionGate("test denorm")
    monkeypatch.setattr(h3_processor, "_DENORM_FUSION", gate)
    shape = (2, 3, 5, 8, 12) if ndim == 5 else (2, 3, 8, 12)
    decoded = torch.randn(shape, device="cuda") * 0.6
    eager = vae.processor._revert_tensor_eager(decoded.clone(), runtime_owned=False)
    with patch.object(
        h3_processor,
        "minimax_h3_vae_denorm_clamp",
        wraps=h3_processor.minimax_h3_vae_denorm_clamp,
    ) as fused:
        fast = vae.processor.revert_tensor(decoded, runtime_owned=runtime_owned)
    # First sight verifies one frame out of place, then the full run follows.
    assert fused.call_count == 2 and gate.verified and not gate.disabled
    assert tuple(fused.call_args_list[0].args[0].shape) == (1, 3, 1, 8, 12)
    assert fast.shape == eager.shape == (2, 3, shape[2] if ndim == 5 else 1, 8, 12)
    assert fast.is_contiguous()
    assert (fast.data_ptr() == decoded.data_ptr()) is runtime_owned
    assert torch.equal(fast, eager)
    assert fast.min().item() >= 0.0 and fast.max().item() <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
