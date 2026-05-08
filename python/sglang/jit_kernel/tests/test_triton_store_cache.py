"""
Tests for triton_store_cache.py — ROCm-only fused FP8 quantise + scatter.

Structure mirrors test_fused_store_index_cache.py (CUDA-side counterpart).

Two test suites:

  flashmla  –  triton_fused_store_flashmla() vs. the two-step fallback
               (quant_to_nope_fp8_rope_bf16_pack_triton + _set_k_and_s_triton_kernel)

  indexer   –  triton_fused_store_indexer() vs. a pure-Python reference
               implementing the same max(1e-8, abs_max)/FP8_MAX quantisation.

Tolerance strategy:
  FP8 has only ~3 decimal digits of precision, so exact bitwise equality is not
  expected after a BF16→FP32→FP8 round-trip with independent scale computation.
  We compare *dequantised* float32 values (fp8_value * scale) with:
    rtol=0.15 (15% relative error) — matches test_fused_store_index_cache.py
    atol=0.5  (half an FP8 step)   — catches systematic scale errors
"""
from __future__ import annotations

import math
import sys
from typing import Tuple

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="nightly-amd-1-gpu", nightly=True)

try:
    from sglang.srt.utils import is_hip

    _IS_HIP = is_hip()
except ImportError:
    _IS_HIP = False

try:
    from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

    _FP8_FNUZ = is_fp8_fnuz()
except ImportError:
    _FP8_FNUZ = False

_FP8_DTYPE = torch.float8_e4m3fnuz if _FP8_FNUZ else torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
_FP8_MIN = float(torch.finfo(_FP8_DTYPE).min)


def _skip_if_not_rocm():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")
    if not _IS_HIP:
        pytest.skip("triton_store_cache kernels are ROCm-only; skipping on CUDA")


# ---------------------------------------------------------------------------
# flashmla helpers
# ---------------------------------------------------------------------------

_NOPE_DIM = 448
_ROPE_DIM = 64
_INPUT_DIM = _NOPE_DIM + _ROPE_DIM  # 512
_TILE_SIZE = 64
_NUM_NOPE_TILES = _NOPE_DIM // _TILE_SIZE  # 7


def _flashmla_bytes_per_page(page_size: int) -> int:
    return math.ceil(584 * page_size / 576) * 576


def _make_flashmla_cache(num_pages: int, page_size: int) -> torch.Tensor:
    bpp = _flashmla_bytes_per_page(page_size)
    return torch.zeros(num_pages, bpp, dtype=torch.uint8, device="cuda")


def _read_flashmla_token(
    cache: torch.Tensor,
    token_idx: int,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract stored data for one token.

    Returns:
        nope_fp8_f32  [448] float32
        rope_bf16     [64]  bfloat16
        scale_ue8m0   [7]   uint8
    """
    bpp = cache.shape[1]
    page = token_idx // page_size
    slot = token_idx % page_size
    flat = cache.reshape(-1)

    # Data region: each slot is 576 bytes (448 FP8 nope + 128 BF16 rope).
    nope_start = page * bpp + slot * 576
    nope_fp8 = flat[nope_start : nope_start + _NOPE_DIM].view(_FP8_DTYPE).float()

    rope_start = page * bpp + slot * 576 + _NOPE_DIM
    rope_bf16 = flat[rope_start : rope_start + _ROPE_DIM * 2].view(torch.bfloat16)

    # Scale region: starts at page_size*576 within the page; 8 bytes per slot.
    s_offset = page * bpp + page_size * 576 + slot * 8
    scale_u8 = flat[s_offset : s_offset + _NUM_NOPE_TILES].clone()

    return nope_fp8, rope_bf16, scale_u8


def _ue8m0_to_scale(ue8m0_bytes: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 bytes to the dequant-time scale: 2^(ue8m0_byte - 127).

    NOTE: do not confuse with the quant-time inv_scale = 2^(127 - ue8m0_byte).
    """
    return torch.exp2(ue8m0_bytes.float() - 127.0)


def _dequant_flashmla_token(
    nope_fp8_f32: torch.Tensor,  # [448] float32
    scale_u8: torch.Tensor,      # [7]   uint8
) -> torch.Tensor:
    scales = _ue8m0_to_scale(scale_u8)
    return nope_fp8_f32 * scales.repeat_interleave(_TILE_SIZE)


# ---------------------------------------------------------------------------
# indexer helpers
# ---------------------------------------------------------------------------

_INDEXER_DIM = 128


def _indexer_bytes_per_page(page_size: int) -> int:
    return 132 * page_size


def _make_indexer_cache(num_pages: int, page_size: int) -> torch.Tensor:
    return torch.zeros(
        num_pages, _indexer_bytes_per_page(page_size), dtype=torch.uint8, device="cuda"
    )


def _read_indexer_token(
    cache: torch.Tensor,
    token_idx: int,
    page_size: int,
) -> Tuple[torch.Tensor, float]:
    """Extract stored data for one token.

    Returns:
        fp8_f32  [128] float32
        scale    float
    """
    bpp = cache.shape[1]
    page = token_idx // page_size
    slot = token_idx % page_size
    flat = cache.reshape(-1)

    fp8_start = page * bpp + slot * 128
    fp8_f32 = flat[fp8_start : fp8_start + 128].view(_FP8_DTYPE).float()

    # Scale region starts at byte page_size*128 within the page.
    scale_start = page * bpp + 128 * page_size + slot * 4
    scale = flat[scale_start : scale_start + 4].view(torch.float32).item()

    return fp8_f32, scale


def _ref_indexer_store(
    input_bf16: torch.Tensor,
    loc: torch.Tensor,
    num_pages: int,
    page_size: int,
) -> torch.Tensor:
    """Pure-Python CPU reference for the indexer store.

    Quantisation: scale = max(1e-8, abs_max) / FP8_MAX; fp8 = clamp(x / scale).
    Runs on CPU to avoid sharing any GPU arithmetic path with the Triton kernel.
    """
    N = input_bf16.shape[0]
    cache = _make_indexer_cache(num_pages, page_size)
    flat = cache.reshape(-1)
    x_f32 = input_bf16.float().cpu()
    loc_cpu = loc.cpu()

    for i in range(N):
        idx = int(loc_cpu[i].item())
        page = idx // page_size
        slot = idx % page_size
        row = x_f32[i]

        scale = max(1e-8, row.abs().max().item()) / _FP8_MAX
        fp8_vals = (row / scale).clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        fp8_start = page * _indexer_bytes_per_page(page_size) + slot * 128
        flat[fp8_start : fp8_start + 128] = fp8_vals.view(torch.uint8).cpu()

        scale_start = (
            page * _indexer_bytes_per_page(page_size) + 128 * page_size + slot * 4
        )
        flat[scale_start : scale_start + 4] = torch.tensor(
            [scale], dtype=torch.float32
        ).view(torch.uint8)

    return cache


# ---------------------------------------------------------------------------
# flashmla tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize(
    "num_tokens,base_index",
    [(1, 0), (32, 0), (64, 0), (128, 64), (257, 65), (512, 0)],
)
def test_flashmla_matches_two_step_fallback(
    num_tokens: int, base_index: int, page_size: int
):
    """Fused kernel must produce the same result as the two-step AMD fallback.

    The two-step fallback is:
        pack = quant_to_nope_fp8_rope_bf16_pack_triton(key)
        _set_k_and_s_triton(cache, loc, pack, page_size)

    Comparison:
      - rope: bitwise equal (no quantisation)
      - UE8M0 scales: within ±1 exponent unit (valid adjacent rounding choices)
      - dequantised nope: rtol=0.15, atol=0.5
    """
    _skip_if_not_rocm()

    from sglang.jit_kernel.triton_store_cache import triton_fused_store_flashmla
    from sglang.srt.layers.attention.nsa.index_buf_accessor_v4 import (
        _set_k_and_s_triton,
    )
    from sglang.srt.layers.attention.nsa.quant_k_cache_v4 import (
        quant_to_nope_fp8_rope_bf16_pack_triton,
    )

    device = torch.device("cuda")
    key = torch.randn((num_tokens, _INPUT_DIM), device=device, dtype=torch.bfloat16)
    loc = (
        base_index + torch.randperm(num_tokens * 2, device=device)[:num_tokens]
    ).to(torch.int64)

    num_pages = int(loc.max().item()) // page_size + 2

    ref_cache = _make_flashmla_cache(num_pages, page_size)
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(key)
    _set_k_and_s_triton(ref_cache, loc, pack, page_size)
    torch.cuda.synchronize()

    fused_cache = _make_flashmla_cache(num_pages, page_size)
    triton_fused_store_flashmla(key, fused_cache, loc, page_size)
    torch.cuda.synchronize()

    for i in range(num_tokens):
        idx = int(loc[i].item())
        ref_nope, ref_rope, ref_scales = _read_flashmla_token(ref_cache, idx, page_size)
        fus_nope, fus_rope, fus_scales = _read_flashmla_token(fused_cache, idx, page_size)

        torch.testing.assert_close(fus_rope, ref_rope, rtol=0, atol=0)

        scale_diff = (fus_scales.int() - ref_scales.int()).abs()
        assert scale_diff.max().item() <= 1, (
            f"token {i}: UE8M0 scale differs by more than 1 exponent unit: {scale_diff}"
        )

        ref_deq = _dequant_flashmla_token(ref_nope, ref_scales)
        fus_deq = _dequant_flashmla_token(fus_nope, fus_scales)
        torch.testing.assert_close(fus_deq, ref_deq, rtol=0.15, atol=0.5)


@pytest.mark.parametrize("page_size", [64, 256])
@pytest.mark.parametrize("num_tokens", [1, 64, 257])
def test_flashmla_roundtrip_reconstruction(num_tokens: int, page_size: int):
    """Dequantised output must approximately reconstruct the original BF16 input."""
    _skip_if_not_rocm()

    from sglang.jit_kernel.triton_store_cache import triton_fused_store_flashmla

    device = torch.device("cuda")
    key = torch.randn((num_tokens, _INPUT_DIM), device=device, dtype=torch.bfloat16)
    loc = torch.arange(num_tokens, device=device, dtype=torch.int64)
    num_pages = num_tokens // page_size + 2

    cache = _make_flashmla_cache(num_pages, page_size)
    triton_fused_store_flashmla(key, cache, loc, page_size)
    torch.cuda.synchronize()

    orig_f32 = key.float()
    for i in range(num_tokens):
        nope_fp8, rope_bf16, scales = _read_flashmla_token(cache, i, page_size)
        torch.testing.assert_close(
            _dequant_flashmla_token(nope_fp8, scales),
            orig_f32[i, :_NOPE_DIM],
            rtol=0.15,
            atol=5e-2,
        )
        torch.testing.assert_close(
            rope_bf16.float(), orig_f32[i, _NOPE_DIM:], rtol=0, atol=0
        )


def test_flashmla_zero_input(page_size: int = 256):
    """Zero input → zero FP8 nope, zero rope. EPS floor keeps scale valid."""
    _skip_if_not_rocm()

    from sglang.jit_kernel.triton_store_cache import triton_fused_store_flashmla

    device = torch.device("cuda")
    key = torch.zeros((4, _INPUT_DIM), device=device, dtype=torch.bfloat16)
    loc = torch.arange(4, device=device, dtype=torch.int64)

    cache = _make_flashmla_cache(1, page_size)
    triton_fused_store_flashmla(key, cache, loc, page_size)
    torch.cuda.synchronize()

    for i in range(4):
        nope_fp8, rope_bf16, _ = _read_flashmla_token(cache, i, page_size)
        assert (nope_fp8 == 0).all(), f"token {i}: expected zero nope FP8 output"
        assert (rope_bf16 == 0).all(), f"token {i}: expected zero rope BF16 output"


# ---------------------------------------------------------------------------
# indexer tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize(
    "num_tokens,base_index",
    [(1, 0), (32, 0), (128, 64), (257, 65)],
)
def test_indexer_matches_reference(num_tokens: int, base_index: int, page_size: int):
    """Fused indexer kernel must match the pure-Python reference.

    Comparison:
      - FP32 scale: near-exact (rtol=1e-4, atol=1e-7)
      - Dequantised FP8 values: rtol=0.15, atol=0.5
    """
    _skip_if_not_rocm()

    from sglang.jit_kernel.triton_store_cache import triton_fused_store_indexer

    device = torch.device("cuda")
    key = torch.randn((num_tokens, _INDEXER_DIM), device=device, dtype=torch.bfloat16)
    loc = (
        base_index + torch.randperm(num_tokens * 2, device=device)[:num_tokens]
    ).to(torch.int64)
    num_pages = int(loc.max().item()) // page_size + 2

    fused_cache = _make_indexer_cache(num_pages, page_size)
    triton_fused_store_indexer(key, fused_cache, loc, page_size)
    torch.cuda.synchronize()

    ref_cache = _ref_indexer_store(key, loc, num_pages, page_size)

    for i in range(num_tokens):
        idx = int(loc[i].item())
        ref_fp8, ref_scale = _read_indexer_token(ref_cache, idx, page_size)
        fus_fp8, fus_scale = _read_indexer_token(fused_cache, idx, page_size)

        torch.testing.assert_close(
            torch.tensor(fus_scale), torch.tensor(ref_scale), rtol=1e-4, atol=1e-7
        )
        torch.testing.assert_close(
            fus_fp8 * fus_scale, ref_fp8 * ref_scale, rtol=0.15, atol=0.5
        )


@pytest.mark.parametrize("num_tokens", [1, 64, 257])
def test_indexer_roundtrip_reconstruction(num_tokens: int, page_size: int = 64):
    """Dequantised indexer output must approximately reconstruct the original input."""
    _skip_if_not_rocm()

    from sglang.jit_kernel.triton_store_cache import triton_fused_store_indexer

    device = torch.device("cuda")
    key = torch.randn((num_tokens, _INDEXER_DIM), device=device, dtype=torch.bfloat16)
    loc = torch.arange(num_tokens, device=device, dtype=torch.int64)
    num_pages = num_tokens // page_size + 2

    cache = _make_indexer_cache(num_pages, page_size)
    triton_fused_store_indexer(key, cache, loc, page_size)
    torch.cuda.synchronize()

    orig_f32 = key.float()
    for i in range(num_tokens):
        fp8, scale = _read_indexer_token(cache, i, page_size)
        torch.testing.assert_close(fp8 * scale, orig_f32[i], rtol=0.15, atol=5e-2)


def test_indexer_zero_input(page_size: int = 64):
    """Zero input → zero FP8 values. EPS prevents divide-by-zero in scale."""
    _skip_if_not_rocm()

    from sglang.jit_kernel.triton_store_cache import triton_fused_store_indexer

    device = torch.device("cuda")
    key = torch.zeros((4, _INDEXER_DIM), device=device, dtype=torch.bfloat16)
    loc = torch.arange(4, device=device, dtype=torch.int64)
    cache = _make_indexer_cache(1, page_size)
    triton_fused_store_indexer(key, cache, loc, page_size)
    torch.cuda.synchronize()

    for i in range(4):
        fp8, _ = _read_indexer_token(cache, i, page_size)
        assert (fp8 == 0).all(), f"token {i}: expected zero FP8 output"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
