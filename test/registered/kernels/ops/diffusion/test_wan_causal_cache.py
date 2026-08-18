"""Wan causal VAE data-movement kernels: the fused conv-input builder and the
fused DupUp3D shortcut add must be bitwise identical to the aten op chains
they replace (they are pure data movement plus zero fill / one fp32 add)."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.triton.wan_causal_cache import (
    cat_pad_channels_last_3d,
    dup_up3d_add,
)
from sglang.multimodal_gen.runtime.models.vaes import wanvae
from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    CACHE_T,
    WanCausalConv3d,
    _cache_payload,
    _run_cached_causal_conv,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _cl3d(shape, dtype):
    return torch.randn(shape, device="cuda", dtype=dtype).contiguous(
        memory_format=torch.channels_last_3d
    )


def _ref_cat_pad(x, cache, padding):
    p = list(padding)
    if cache is not None:
        x = torch.cat([cache, x], dim=2)
        p[4] -= cache.shape[2]
    if any(p):
        x = F.pad(x, p)
    return x.contiguous(memory_format=torch.channels_last_3d)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "c,t,h,w,cache_t,pads",
    [
        (96, 1, 10, 14, 0, (1, 1, 1, 1, 2, 0)),  # first chunk, zero-fill front
        (96, 1, 10, 14, 1, (1, 1, 1, 1, 2, 0)),  # legacy 1-frame cache
        (96, 1, 10, 14, 2, (1, 1, 1, 1, 2, 0)),  # steady state k3 conv
        (64, 1, 10, 14, 2, (0, 0, 0, 0, 2, 0)),  # time_conv (temporal only)
        (48, 4, 10, 14, 2, (1, 1, 1, 1, 2, 0)),  # encoder-style T=4 chunk
    ],
)
def test_cat_pad_bitwise(dtype, c, t, h, w, cache_t, pads) -> None:
    torch.cuda.manual_seed(0)
    x = _cl3d((1, c, t, h, w), dtype)
    cache = None
    if cache_t:
        # Strided interior view: caches may arrive as non-contiguous slices.
        ph, pw = pads[2], pads[0]
        buf = _cl3d((1, c, cache_t, h + 2 * ph, w + 2 * pw), dtype)
        cache = buf[:, :, :, ph : ph + h, pw : pw + w]
    out = cat_pad_channels_last_3d(x, cache, pads)
    ref = _ref_cat_pad(x, cache, pads)
    assert out is not None and out.shape == ref.shape
    assert out.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(out, ref)

    # Dual-output mode: the same pass also emits the compact feature cache
    # (unpadded interior of the last frames), bitwise equal to the slice.
    pair = cat_pad_channels_last_3d(x, cache, pads, keep_cache_t=2)
    assert pair is not None
    out2, keep = pair
    assert torch.equal(out2, ref)
    ph, pw = pads[2], pads[0]
    keep_t = min(2, ref.shape[2])
    want = ref[:, :, ref.shape[2] - keep_t :, ph : ph + h, pw : pw + w]
    assert keep.shape == want.shape
    assert keep.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(keep, want)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "c_in,c_out,t,h,w,ft,fs,drop",
    [
        (128, 64, 1, 10, 14, 2, 2, False),
        (128, 64, 1, 10, 14, 2, 2, True),  # first_chunk slicing
        (64, 32, 2, 10, 14, 1, 2, False),
    ],
)
def test_dup_up3d_add_bitwise(dtype, c_in, c_out, t, h, w, ft, fs, drop) -> None:
    torch.cuda.manual_seed(0)
    repeats = c_out * ft * fs * fs // c_in
    src = _cl3d((1, c_in, t, h, w), dtype)
    t_out = t * ft - (ft - 1 if drop else 0)
    # Main arm as a permuted view, like the WanResample 2D output.
    main = torch.randn(
        (1, t_out, c_out, h * fs, w * fs), device="cuda", dtype=dtype
    ).permute(0, 2, 1, 3, 4)

    dup = src.repeat_interleave(repeats, dim=1)
    dup = dup.view(1, c_out, ft, fs, fs, t, h, w)
    dup = dup.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    dup = dup.view(1, c_out, t * ft, h * fs, w * fs)
    if drop:
        dup = dup[:, :, ft - 1 :, :, :]
    ref = main + dup

    out = dup_up3d_add(main, src, ft, fs, repeats, drop)
    assert out is not None and out.shape == ref.shape
    # Layout must match the aten add output exactly (downstream reductions
    # are layout-sensitive), and every value must be bitwise identical.
    assert out.stride() == ref.stride()
    assert torch.equal(out, ref)


@torch.no_grad()
@pytest.mark.parametrize("pads_temporal_only", [False, True])
def test_cached_conv_chunk_loop_bitwise(pads_temporal_only) -> None:
    """The fused conv-input/compact-cache scheme must reproduce the original
    clone/cat bookkeeping bitwise across a chunked decode, including the
    first-chunk zero fill and the "Rep" marker start used by WanResample."""
    torch.cuda.manual_seed(0)
    c = 64
    if pads_temporal_only:
        conv = WanCausalConv3d(c, 2 * c, (3, 1, 1), padding=(1, 0, 0))
    else:
        conv = WanCausalConv3d(c, c, 3, padding=1)
    conv = conv.to(device="cuda", dtype=torch.float32)
    conv.weight.data = conv.weight.data.contiguous(memory_format=torch.channels_last_3d)
    chunks = [_cl3d((1, c, 1, 10, 14), torch.float32) for _ in range(4)]

    def run(force_fallback, start):
        cache = [start]
        outs = []
        if force_fallback:
            orig = wanvae.cat_pad_channels_last_3d
            wanvae.cat_pad_channels_last_3d = None
        try:
            for x in chunks:
                outs.append(_run_cached_causal_conv(conv, x, cache, 0))
        finally:
            if force_fallback:
                wanvae.cat_pad_channels_last_3d = orig
        return outs, cache[0]

    for start in (None, "Rep"):
        fused_outs, fused_cache = run(False, start)
        ref_outs, ref_cache = run(True, start)
        for got, want in zip(fused_outs, ref_outs, strict=True):
            assert torch.equal(got, want)
        got_payload = _cache_payload(fused_cache)
        assert got_payload is not None and got_payload.shape[2] == CACHE_T
        # Reference cache holds the last CACHE_T unpadded frames.
        assert torch.equal(got_payload, ref_cache[:, :, -CACHE_T:])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
