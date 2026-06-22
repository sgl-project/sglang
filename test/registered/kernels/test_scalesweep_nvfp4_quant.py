# SPDX-License-Identifier: Apache-2.0
"""Kernel-level tests for ScaleSweep NVFP4 quantization (sgl-project/sglang#27246).

Validates the ported Triton ScaleSweep kernel against a PyTorch reference and
checks that it reduces FP4 reconstruction error versus the absmax baseline.
Native FP4 requires SM100+ (Blackwell).
"""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("NVFP4 tests require CUDA.", allow_module_level=True)

from sglang.srt.layers.quantization.scalesweep_nvfp4.scalesweep_mse_nvfp4_quant import (
    BLOCK_SIZE,
    FP4_E2M1_MAX,
    LOWER_BOUND,
    REF_MAX_SCALE_RAW,
    UPPER_BOUND,
    create_fp4_output_tensors,
    round_up,
    scalesweep_mse_nvfp4_quant,
)

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(1, 16), (3, 64), (32, 128), (128, 64), (150, 80)]
# SGLang's NVFP4 block-scale convention (matches `448 * 6 / amax`).
FP8_BLOCK_SCALE_MAX = 448.0

E2M1_TO_FLOAT32 = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _cast_from_fp4(x, m, n):
    lut = torch.tensor(E2M1_TO_FLOAT32, device=x.device, dtype=torch.float32)
    c = torch.stack((x & 0xF, (x >> 4) & 0xF), dim=-1).long()
    return lut[c].reshape(m, n)


def _cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    out = torch.empty_like(x)
    out[(x >= 0.0) & (x <= 0.25)] = 0.0
    out[(x > 0.25) & (x < 0.75)] = 0.5
    out[(x >= 0.75) & (x <= 1.25)] = 1.0
    out[(x > 1.25) & (x < 1.75)] = 1.5
    out[(x >= 1.75) & (x <= 2.5)] = 2.0
    out[(x > 2.5) & (x < 3.5)] = 3.0
    out[(x >= 3.5) & (x <= 5.0)] = 4.0
    out[x > 5.0] = 6.0
    return out * sign


def _global_scale_inv(x):
    amax = torch.abs(x).max().to(torch.float32)
    return FP8_BLOCK_SCALE_MAX * FP4_E2M1_MAX / amax


def _recover_swizzled_scales(scale, m, n):
    scale_n = n // BLOCK_SIZE
    rounded_m, rounded_n = round_up(m, 128), round_up(scale_n, 4)
    tmp = scale.reshape(1, rounded_m // 128, rounded_n // 4, 32, 4, 4)
    tmp = tmp.permute(0, 1, 4, 3, 2, 5)
    return tmp.reshape(rounded_m, rounded_n).to(torch.float32)[:m, :scale_n]


def _ref_quant(x, global_scale_inv, sweep):
    """PyTorch reference. ``sweep=False`` is the absmax baseline (offset 0 only)."""
    m, n = x.shape
    blocks = x.reshape(m, n // BLOCK_SIZE, BLOCK_SIZE).to(torch.float32) * global_scale_inv
    abs_max = blocks.abs().amax(dim=-1)
    base_raw = (abs_max / FP4_E2M1_MAX).to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
    lo, hi = (LOWER_BOUND, UPPER_BOUND) if sweep else (0, 0)
    offsets = torch.arange(lo, hi + 1, device=x.device, dtype=torch.int32)
    scale_raw = torch.clamp(base_raw.unsqueeze(-1) + offsets, 1, REF_MAX_SCALE_RAW).to(torch.uint8)
    scales = scale_raw.view(torch.float8_e4m3fn).to(torch.float32)
    scaled = blocks.unsqueeze(2) / scales.unsqueeze(-1)
    quantized = _cast_to_fp4(scaled)
    recon = quantized * scales.unsqueeze(-1)
    sq_err = ((recon - blocks.unsqueeze(2)) ** 2).sum(dim=-1)
    best = torch.argmin(sq_err, dim=-1)
    best_scale = torch.gather(scale_raw, 2, best.unsqueeze(-1)).squeeze(-1).view(torch.float8_e4m3fn)
    best_q = torch.gather(
        quantized, 2, best[:, :, None, None].expand(-1, -1, 1, BLOCK_SIZE)
    ).squeeze(2)
    mse = torch.gather(sq_err, 2, best.unsqueeze(-1)).mean().item() / BLOCK_SIZE
    return best_q.reshape(m, n), best_scale, mse


def _require_blackwell():
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(f"native FP4 requires SM100+, got {major}.{minor}")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@torch.inference_mode()
def test_scalesweep_matches_reference(dtype, shape, is_sf_swizzled_layout):
    _require_blackwell()
    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda:0", dtype=dtype)
    gsi = _global_scale_inv(x)

    out_ref, scale_ref, _ = _ref_quant(x, gsi, sweep=True)
    out, out_scale = scalesweep_mse_nvfp4_quant(
        x, gsi, is_sf_swizzled_layout=is_sf_swizzled_layout
    )
    exp_out, exp_scale = create_fp4_output_tensors(
        shape[0], shape[1], torch.device("cuda:0"), is_sf_swizzled_layout
    )

    assert out.shape == exp_out.shape and out.dtype == torch.uint8
    assert out_scale.shape == exp_scale.shape and out_scale.dtype == torch.float8_e4m3fn

    out_vals = _cast_from_fp4(out, *shape)
    scale_vals = (
        _recover_swizzled_scales(out_scale, *shape)
        if is_sf_swizzled_layout
        else out_scale.to(torch.float32)
    )
    torch.testing.assert_close(out_vals, out_ref)
    torch.testing.assert_close(scale_vals, scale_ref.to(torch.float32))


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_scalesweep_beats_absmax_mse(dtype):
    _require_blackwell()
    torch.manual_seed(0)
    x = torch.randn(2048, 4096, device="cuda:0", dtype=dtype)
    gsi = _global_scale_inv(x)
    _, _, mse_sweep = _ref_quant(x, gsi, sweep=True)
    _, _, mse_absmax = _ref_quant(x, gsi, sweep=False)
    assert mse_sweep < mse_absmax, (mse_sweep, mse_absmax)
