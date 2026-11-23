"""``pixel_norm_silu``: bit-exact per-pixel RMS norm + SiLU into a conv input.

Oracle: aten itself. The eager chain is
``silu(x / sqrt(mean(x**2, dim=1, keepdim=True) + eps))`` in fp32 (autocast
promotes ``pow`` on a half-precision ``x``), then one cast to the conv dtype.
The kernel keeps aten's ``mean`` and must reproduce every other bit, for
channels_last_3d and NCDHW inputs, into a channels_last destination view.
"""

import sys

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.numerics import cuda_expf, silu_f32
from sglang.kernels.ops.diffusion import can_use_pixel_norm_silu, pixel_norm_silu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

DEVICE = "cuda"
EPS = 1e-8
HALF = (torch.bfloat16, torch.float16)


@triton.jit
def _numerics_kernel(x_ptr, out_ptr, n, SILU: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if SILU:
        y = silu_f32(x)
    else:
        y = cuda_expf(x)
    tl.store(out_ptr + offs, y, mask=mask)


def _input(shape, dtype, channels_last, scale=1.0):
    x = (torch.randn(shape, device=DEVICE) * scale).to(dtype)
    if channels_last:
        return x.contiguous(memory_format=torch.channels_last_3d)
    return x.contiguous()


def _reference(x, out_dtype, eps=EPS):
    xf = x.float()
    rms = torch.sqrt(torch.mean(xf**2, dim=1, keepdim=True) + eps)
    return F.silu(xf / rms).to(out_dtype)


def _bits(t):
    return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def _dest(x, out_dtype, left=0, right=0):
    b, c, t, h, w = x.shape
    padded = torch.full(
        (b, c, t + left + right, h, w), 7.0, device=DEVICE, dtype=out_dtype
    ).contiguous(memory_format=torch.channels_last_3d)
    return padded, padded[:, :, left : left + t]


@torch.no_grad()
@pytest.mark.parametrize(
    "x_dtype,out_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
@pytest.mark.parametrize("channels_last", [True, False])
@pytest.mark.parametrize(
    "shape",
    [(1, 128, 3, 16, 24), (2, 24, 2, 5, 7), (1, 1024, 2, 6, 6), (1, 3, 4, 8, 8)],
)
def test_pixel_norm_silu_matches_aten(x_dtype, out_dtype, channels_last, shape):
    x = _input(shape, x_dtype, channels_last)
    padded, out = _dest(x, out_dtype)
    assert can_use_pixel_norm_silu(x, EPS, out)
    assert pixel_norm_silu(x, EPS, out) is out
    ref = _reference(x, out_dtype)
    assert torch.equal(_bits(out), _bits(ref))


@torch.no_grad()
@pytest.mark.parametrize("dtype", HALF)
@pytest.mark.parametrize("channels_last", [True, False])
def test_pixel_norm_silu_matches_autocast_module_chain(dtype, channels_last):
    # The LTX VAE chain as it runs under autocast, cast as the conv casts it.
    x = _input((1, 96, 5, 12, 20), dtype, channels_last)
    with torch.autocast("cuda", dtype=dtype):
        eager = F.silu(x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + EPS))
    assert eager.dtype == torch.float32
    _, out = _dest(x, dtype)
    pixel_norm_silu(x, EPS, out)
    assert torch.equal(_bits(out), _bits(eager.to(dtype)))


@torch.no_grad()
@pytest.mark.parametrize("scale", [0.0, 1e-6, 1e-3, 30.0, 3e3])
def test_pixel_norm_silu_value_ranges(scale):
    x = _input((1, 64, 2, 9, 11), torch.bfloat16, True, scale=scale)
    x[0, :, 0, 0, 0] = 0.0
    x[0, :, 0, 0, 1] = -0.0
    for out_dtype in (torch.bfloat16, torch.float32):
        _, out = _dest(x, out_dtype)
        pixel_norm_silu(x, EPS, out)
        assert torch.equal(_bits(out), _bits(_reference(x, out_dtype)))


@torch.no_grad()
@pytest.mark.parametrize("silu", [False, True])
def test_cuda_expf_and_silu_match_aten(silu):
    # aten's CUDA exp / SiLU call libdevice expf; tl.exp is an approximation.
    n = 1 << 20
    every_bf16 = torch.arange(-32768, 32768, device=DEVICE).to(torch.int16)
    x = torch.cat(
        [
            torch.randn(n, device=DEVICE) * 3,
            (torch.rand(n, device=DEVICE) - 0.5) * 400,
            torch.randn(n, device=DEVICE) * 1e-4,
            every_bf16.view(torch.bfloat16).float().nan_to_num(0.0, 0.0, 0.0),
            torch.tensor(
                [0.0, -0.0, 88.7, -88.7, 104.0, -104.0, float("inf"), float("-inf")],
                device=DEVICE,
            ),
        ]
    )
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), 1024),)
    _numerics_kernel[grid](x, out, x.numel(), SILU=silu, BLOCK=1024)
    ref = F.silu(x) if silu else torch.exp(x)
    assert torch.equal(_bits(out), _bits(ref))


@torch.no_grad()
def test_pixel_norm_silu_writes_only_the_body_of_a_padded_buffer():
    x = _input((2, 48, 3, 6, 10), torch.bfloat16, False)
    padded, out = _dest(x, torch.bfloat16, left=2, right=1)
    assert can_use_pixel_norm_silu(x, EPS, out)
    pixel_norm_silu(x, EPS, out)
    assert torch.equal(_bits(out), _bits(_reference(x, torch.bfloat16)))
    assert torch.all(padded[:, :, :2] == 7.0) and torch.all(padded[:, :, 5:] == 7.0)
    assert padded.is_contiguous(memory_format=torch.channels_last_3d)


@torch.no_grad()
def test_pixel_norm_silu_rejects_unsupported_inputs():
    x = _input((1, 32, 2, 6, 6), torch.bfloat16, True)
    _, out = _dest(x, torch.bfloat16)
    assert can_use_pixel_norm_silu(x, EPS, out)
    assert not can_use_pixel_norm_silu(x, EPS, out.contiguous())  # NCDHW destination
    assert not can_use_pixel_norm_silu(x, EPS, out[:, :, :1])  # shape
    assert not can_use_pixel_norm_silu(x, EPS, out.cpu())  # device
    assert not can_use_pixel_norm_silu(x.cpu(), EPS, out)
    assert not can_use_pixel_norm_silu(x.double(), EPS, out)  # dtype
    assert not can_use_pixel_norm_silu(x[:, :, :, ::2], EPS, out[:, :, :, ::2])
    assert not can_use_pixel_norm_silu(x[:, :0], EPS, out[:, :0])  # empty
    x1 = torch.randn(1, 1, 2, 3, 4, device=DEVICE, dtype=torch.bfloat16)
    assert not can_use_pixel_norm_silu(x1, EPS, torch.empty_like(x1))  # C == 1
    with torch.enable_grad():
        xg = x.float().requires_grad_(True)
        assert not can_use_pixel_norm_silu(xg, EPS, out)
    with pytest.raises(ValueError):
        pixel_norm_silu(x, EPS, out.contiguous())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
