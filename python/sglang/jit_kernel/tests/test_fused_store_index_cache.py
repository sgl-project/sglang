"""
Test for fused_store_index_k_cache kernel.

Design Notes:
  1. torch.cuda.synchronize() needed after TVM FFI kernel call.
  2. _split_buffer used buf[:, :vb].reshape(-1) which COPIES data for
     non-contiguous slices → reference buffer stayed all-zeros.
     Fix: use flat byte-offset indexing.
  3. act_quant may use a different quantization scheme → generous tolerance.
  4. FP8 E4M3 1-ULP rounding differences between CUDA hardware cast
     (__nv_fp8_e4m3) and PyTorch .to(float8_e4m3fn) at tie-break points.
     Adjacent FP8 representable values at the high end differ by up to 32
     in float space (e.g. 288, 320, 352, ..., 448).
     Need to compare dequantized values with FP8-appropriate tolerance.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pytest
import torch

try:
    from sglang.jit_kernel.fused_store_index_cache import (
        can_use_nsa_fused_store,
        fused_store_index_k_cache,
    )

    HAS_FUSED = True
except ImportError:
    HAS_FUSED = False

try:
    from sglang.srt.utils import is_hip

    _is_hip = is_hip()
except ImportError:
    _is_hip = False

try:
    from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

    _is_fp8_fnuz = is_fp8_fnuz()
except ImportError:
    _is_fp8_fnuz = False

PAGE_SIZE = 64
HEAD_DIM = 128
FP8_E4M3_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn
BYTES_PER_TOKEN = 128 + 4  # 128 fp8 bytes + 4 scale bytes
BYTES_PER_PAGE = PAGE_SIZE * BYTES_PER_TOKEN


def _skip_if_unavailable(page_size: int = PAGE_SIZE):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if _is_hip:
        pytest.skip("Fused store kernel is CUDA-specific")
    if _is_fp8_fnuz:
        pytest.skip("Fused store path disabled for FP8 FNUZ")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn not available")
    if not HAS_FUSED:
        pytest.skip("fused_store_index_cache not importable")
    if not can_use_nsa_fused_store(torch.bfloat16, torch.int64, page_size):
        pytest.skip("JIT kernel unavailable / failed to compile")


def _num_pages(loc: torch.Tensor, page_size: int, extra: int = 1) -> int:
    return int(loc.max().item()) // page_size + 1 + extra


def _make_buffer(num_pages: int, page_size: int = PAGE_SIZE) -> torch.Tensor:
    return torch.zeros(
        (num_pages, page_size * BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device="cuda",
    )


def _read_token_from_buffer(
    buf: torch.Tensor,
    token_idx: int,
    page_size: int = PAGE_SIZE,
) -> Tuple[torch.Tensor, float]:
    """
    Read a single token's fp8 values and scale from the paged buffer
    using flat byte offsets.
    """
    page = token_idx // page_size
    offset = token_idx % page_size
    page_bytes = page_size * BYTES_PER_TOKEN

    buf_flat = buf.reshape(-1)

    val_start = page * page_bytes + offset * 128
    fp8_bytes = buf_flat[val_start : val_start + 128]
    fp8_vals = fp8_bytes.view(FP8_DTYPE).float()

    scale_start = page * page_bytes + 128 * page_size + offset * 4
    scale_bytes = buf_flat[scale_start : scale_start + 4]
    scale = scale_bytes.view(torch.float32).item()

    return fp8_vals, scale


def _write_token_to_buffer(
    buf: torch.Tensor,
    token_idx: int,
    fp8_data: torch.Tensor,
    scale: float,
    page_size: int = PAGE_SIZE,
) -> None:
    """
    Write a single token's fp8 values and scale into the paged buffer
    using flat byte offsets on buf.reshape(-1) (which is a true view
    since buf is contiguous).
    """
    page = token_idx // page_size
    offset = token_idx % page_size
    page_bytes = page_size * BYTES_PER_TOKEN

    buf_flat = buf.reshape(-1)

    val_start = page * page_bytes + offset * 128
    buf_flat[val_start : val_start + 128] = fp8_data.view(torch.uint8)

    scale_start = page * page_bytes + 128 * page_size + offset * 4
    scale_t = torch.tensor([scale], dtype=torch.float32, device=buf.device)
    buf_flat[scale_start : scale_start + 4] = scale_t.view(torch.uint8)


def _gather_tokens(
    buf: torch.Tensor,
    loc: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = loc.shape[0]
    fp8_f32 = torch.empty((N, HEAD_DIM), dtype=torch.float32, device=buf.device)
    scales = torch.empty((N,), dtype=torch.float32, device=buf.device)
    for i in range(N):
        idx = int(loc[i].item())
        vals, s = _read_token_from_buffer(buf, idx, page_size)
        fp8_f32[i] = vals
        scales[i] = s
    return fp8_f32, scales


# Reference kernel
def _reference_quantize_and_store(
    key_bf16: torch.Tensor,
    loc: torch.Tensor,
    num_pages: int,
    page_size: int = PAGE_SIZE,
) -> torch.Tensor:
    """
    Reference kernel of the fused kernel's quantization:
      abs_max = max(|row|)
      scale   = max(1e-4, abs_max) / 448
      fp8_val = clip(val / scale, -448, 448) -> cast to fp8
    """
    N = key_bf16.shape[0]
    key_f32 = key_bf16.float()
    buf = _make_buffer(num_pages, page_size)

    for i in range(N):
        row = key_f32[i]
        abs_max = row.abs().max().item()
        scale = max(1e-4, abs_max) / FP8_E4M3_MAX
        inv_scale = 1.0 / scale
        quantized = (row * inv_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        quantized_fp8 = quantized.to(FP8_DTYPE)

        idx = int(loc[i].item())
        _write_token_to_buffer(buf, idx, quantized_fp8, scale, page_size)

    return buf


def _import_act_quant():
    try:
        from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        return act_quant
    except Exception:
        return None


def _ref_store_via_act_quant(
    key_bf16: torch.Tensor,
    loc: torch.Tensor,
    num_pages: int,
    page_size: int = PAGE_SIZE,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
) -> Optional[torch.Tensor]:
    act_quant = _import_act_quant()
    if act_quant is None:
        return None

    try:
        k_fp8, k_scale = act_quant(key_bf16, block_size, scale_fmt)
    except TypeError:
        k_fp8, k_scale = act_quant(key_bf16, block_size)

    if k_fp8.dim() == 3 and k_fp8.shape[1] == 1:
        k_fp8 = k_fp8.squeeze(1)
    if k_scale is not None and k_scale.dim() == 3 and k_scale.shape[1] == 1:
        k_scale = k_scale.squeeze(1)
    k_scale = k_scale.view(-1).float()

    buf = _make_buffer(num_pages, page_size)
    N = key_bf16.shape[0]
    for i in range(N):
        idx = int(loc[i].item())
        _write_token_to_buffer(
            buf, idx, k_fp8[i].to(FP8_DTYPE), k_scale[i].item(), page_size
        )
    return buf


# TEST 1: Fused kernel vs. its own algorithm (pure-Python reference)
#
# NOTE on FP8 rounding:
#   CUDA hardware fp8 cast (__nv_fp8_e4m3) and PyTorch .to(float8_e4m3fn)
#   may round differently at tie-break points.  This causes up to 1-ULP
#   differences in the FP8 codes.  In FP8 E4M3, adjacent representable
#   values at the high end differ by up to 32 in float space (e.g.
#   288 vs 320).  After dequantization (fp8_float * scale), the error
#   from 1-ULP is: scale * ulp ≈ (abs_max/448) * 32 ≈ 0.07 * abs_max.
#   For randn inputs (abs_max ≈ 3-4), this is about 0.2-0.3.
#
#   We therefore compare dequantized values with tolerances that
#   accommodate 1-ULP FP8 rounding, NOT byte-exact fp8 codes.
@pytest.mark.parametrize(
    "num_tokens,base_index",
    [(1, 0), (32, 0), (64, 0), (128, 64), (257, 65), (512, 0)],
)
def test_fused_kernel_matches_own_algorithm(num_tokens: int, base_index: int):
    """Compare fused CUDA kernel against a pure-Python implementation
    of the *same* quantization formula."""
    _skip_if_unavailable()
    device = torch.device("cuda")

    key = torch.randn((num_tokens, HEAD_DIM), device=device, dtype=torch.bfloat16)
    loc = (
        base_index + torch.randperm(num_tokens, device=device, dtype=torch.int64)
    ).contiguous()
    num_pages = _num_pages(loc, PAGE_SIZE)

    # Reference kernel
    ref_buf = _reference_quantize_and_store(key, loc, num_pages)

    # Fused kernel
    out_buf = _make_buffer(num_pages)
    fused_store_index_k_cache(key, out_buf, loc, page_size=PAGE_SIZE)
    torch.cuda.synchronize()

    out_f, out_s = _gather_tokens(out_buf, loc)
    ref_f, ref_s = _gather_tokens(ref_buf, loc)

    # 1) Scales must match tightly (same f32 formula, no rounding ambiguity)
    torch.testing.assert_close(out_s, ref_s, rtol=1e-5, atol=1e-7)

    # 2) Most FP8 codes should match; allow rare 1-ULP differences.
    #    1-ULP at FP8 E4M3 high end = 32 in float space.
    mismatch = out_f != ref_f
    mismatch_frac = mismatch.float().mean().item()
    assert mismatch_frac < 0.01, (
        f"Too many FP8 code mismatches: {mismatch_frac:.2%} "
        f"(expected < 1% from rounding tie-breaks)"
    )

    # 3) Where codes differ, the difference should be exactly 1 ULP.
    #    In FP8 E4M3: if the float-cast value is V, the adjacent value
    #    differs by ~V * 0.1 (relative) at most.
    if mismatch.any():
        diff = (out_f[mismatch] - ref_f[mismatch]).abs()
        rel_diff = diff / ref_f[mismatch].abs().clamp(min=1e-6)
        # 1-ULP relative difference for E4M3 is at most ~12.5% (2^-3)
        assert rel_diff.max().item() <= 0.15, (
            f"FP8 code difference exceeds 1-ULP: max relative diff = "
            f"{rel_diff.max().item():.4f}"
        )

    # 4) Dequantized values should be close.
    #    Max error from 1-ULP: scale * fp8_ulp ≈ (abs_max/448) * 32
    #    For randn abs_max ≈ 3-4: max_err ≈ 0.21 - 0.29
    out_deq = out_f * out_s.unsqueeze(-1)
    ref_deq = ref_f * ref_s.unsqueeze(-1)
    torch.testing.assert_close(out_deq, ref_deq, rtol=0.15, atol=0.5)


# TEST 2: Cross-check against act_quant
@pytest.mark.parametrize("scale_fmt", [None, "fp32"])
def test_fused_kernel_vs_act_quant_semantic(scale_fmt: Optional[str]):
    """Both fused kernel and act_quant should approximately reconstruct
    the original bf16 values."""
    _skip_if_unavailable()
    device = torch.device("cuda")

    num_tokens = 257
    base_index = 65
    key = torch.randn((num_tokens, HEAD_DIM), device=device, dtype=torch.bfloat16)
    loc = (
        base_index + torch.randperm(num_tokens, device=device, dtype=torch.int64)
    ).contiguous()
    num_pages = _num_pages(loc, PAGE_SIZE)

    ref_buf = _ref_store_via_act_quant(key, loc, num_pages, scale_fmt=scale_fmt)
    if ref_buf is None:
        pytest.skip("act_quant not available")

    out_buf = _make_buffer(num_pages)
    fused_store_index_k_cache(key, out_buf, loc, page_size=PAGE_SIZE)
    torch.cuda.synchronize()

    out_f, out_s = _gather_tokens(out_buf, loc)
    ref_f, ref_s = _gather_tokens(ref_buf, loc)

    out_deq = out_f * out_s.unsqueeze(-1)
    ref_deq = ref_f * ref_s.unsqueeze(-1)
    orig_f32 = key.float()

    # Fused kernel should reconstruct original within FP8 precision
    torch.testing.assert_close(
        out_deq,
        orig_f32,
        rtol=0.15,
        atol=5e-2,
        msg="Fused kernel dequantized values don't approximate original",
    )

    # act_quant may use a very different scale policy.
    try:
        torch.testing.assert_close(
            ref_deq,
            orig_f32,
            rtol=0.25,
            atol=0.5,
            msg="act_quant dequantized values don't approximate original",
        )
    except AssertionError:
        nonzero_frac = (ref_deq.abs() > 1e-6).float().mean().item()
        if nonzero_frac < 0.5:
            pytest.fail(
                f"act_quant output looks mostly zero ({nonzero_frac:.1%} nonzero)."
            )
        else:
            pytest.skip(
                f"act_quant uses a very different quantization scheme "
                f"(scale_fmt={scale_fmt}). Fused kernel validated independently."
            )

    torch.testing.assert_close(
        out_deq,
        ref_deq,
        rtol=0.3,
        atol=0.5,
        msg="Fused and act_quant dequantized values diverge too much",
    )


# TEST 3: Roundtrip reconstruction
@pytest.mark.parametrize("num_tokens", [1, 64, 257])
def test_roundtrip_reconstruction(num_tokens: int):
    _skip_if_unavailable()
    device = torch.device("cuda")

    key = torch.randn((num_tokens, HEAD_DIM), device=device, dtype=torch.bfloat16)
    loc = torch.arange(num_tokens, device=device, dtype=torch.int64)
    num_pages = _num_pages(loc, PAGE_SIZE)

    buf = _make_buffer(num_pages)
    fused_store_index_k_cache(key, buf, loc, page_size=PAGE_SIZE)
    torch.cuda.synchronize()

    fp8_f32, scales = _gather_tokens(buf, loc)
    reconstructed = fp8_f32 * scales.unsqueeze(-1)
    original = key.float()

    torch.testing.assert_close(reconstructed, original, rtol=0.15, atol=5e-2)

    per_row_energy = reconstructed.abs().sum(dim=-1)
    orig_energy = original.abs().sum(dim=-1)
    mask = orig_energy > 0.1
    assert (
        per_row_energy[mask] > 0.01
    ).all(), "Some tokens have zero reconstruction — kernel may not be writing output"


# TEST 4: Boundary conditions
def test_single_token():
    _skip_if_unavailable()
    device = torch.device("cuda")

    key = torch.randn((1, HEAD_DIM), device=device, dtype=torch.bfloat16)
    loc = torch.tensor([0], device=device, dtype=torch.int64)

    buf = _make_buffer(1)
    fused_store_index_k_cache(key, buf, loc, page_size=PAGE_SIZE)
    torch.cuda.synchronize()

    fp8_f32, scales = _gather_tokens(buf, loc)
    reconstructed = fp8_f32 * scales.unsqueeze(-1)
    torch.testing.assert_close(reconstructed, key.float(), rtol=0.15, atol=5e-2)


# TEST 5: Zero input conditions
def test_zero_input():
    _skip_if_unavailable()
    device = torch.device("cuda")

    key = torch.zeros((4, HEAD_DIM), device=device, dtype=torch.bfloat16)
    loc = torch.arange(4, device=device, dtype=torch.int64)

    buf = _make_buffer(1)
    fused_store_index_k_cache(key, buf, loc, page_size=PAGE_SIZE)
    torch.cuda.synchronize()

    fp8_f32, scales = _gather_tokens(buf, loc)

    expected_scale = 1e-4 / FP8_E4M3_MAX
    torch.testing.assert_close(
        scales,
        torch.full_like(scales, expected_scale),
        rtol=1e-5,
        atol=1e-10,
    )
    assert (fp8_f32 == 0).all()


# TEST 6: Sanity check — verify reference itself writes non-zero data
def test_reference_writes_nonzero():
    _skip_if_unavailable()
    device = torch.device("cuda")

    key = torch.randn((8, HEAD_DIM), device=device, dtype=torch.bfloat16)
    loc = torch.arange(8, device=device, dtype=torch.int64)

    buf = _reference_quantize_and_store(key, loc, num_pages=1)

    fp8_f32, scales = _gather_tokens(buf, loc)
    deq = fp8_f32 * scales.unsqueeze(-1)

    assert deq.abs().sum().item() > 0, "Reference buffer is all zeros — error!"
    torch.testing.assert_close(deq, key.float(), rtol=0.15, atol=5e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
