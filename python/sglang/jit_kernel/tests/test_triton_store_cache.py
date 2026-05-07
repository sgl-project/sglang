"""
Tests for triton_store_cache.py — ROCm-only fused FP8 quantise + scatter.

Structure mirrors test_fused_store_index_cache.py (CUDA-side counterpart).

Two test suites:

  flashmla  –  triton_fused_store_flashmla() vs. the two-step fallback
               (quant_to_nope_fp8_rope_bf16_pack_triton + _set_k_and_s_triton_kernel)
               Running both paths on the same input verifies the fused kernel
               produces bit-equivalent (within FP8 ULP) results.

  indexer   –  triton_fused_store_indexer() vs. a pure-Python reference
               implementing the same max(1e-8, abs_max)/FP8_MAX quantisation.
               The Python reference gives us a clearly correct ground truth.

Tolerance strategy:
  FP8 has only ~3 decimal digits of precision, so exact bitwise equality is not
  expected after a BF16→FP32→FP8 round-trip with independent scale computation.
  We compare *dequantised* float32 values (fp8_value * inv_scale) with:
    rtol=0.15 (15% relative error) — matches test_fused_store_index_cache.py
    atol=0.5  (half an FP8 step)   — catches systematic scale errors
  This is the same tolerance scheme used in the CUDA-side counterpart.
"""
from __future__ import annotations

import sys
from typing import Tuple

import pytest
import torch

# ---------------------------------------------------------------------------
# Runtime capability flags
# ---------------------------------------------------------------------------

# Detect whether sglang thinks we are on ROCm/HIP.
# Importing is guarded so tests can be *collected* on CUDA machines (they will
# be skipped at runtime via _skip_if_not_rocm()).
try:
    from sglang.srt.utils import is_hip

    _IS_HIP = is_hip()
except ImportError:
    _IS_HIP = False

# Detect which FP8 variant is in use.  ROCm uses float8_e4m3fnuz (FNUZ);
# CUDA uses float8_e4m3fn.  The choice affects FP8_MAX and FP8_MIN constants.
try:
    from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

    _FP8_FNUZ = is_fp8_fnuz()
except ImportError:
    _FP8_FNUZ = False

# Select the concrete FP8 dtype based on the hardware detected above.
_FP8_DTYPE = (
    torch.float8_e4m3fnuz if _FP8_FNUZ else torch.float8_e4m3fn
)
# Pre-compute the FP8 clamp bounds as plain floats for Python-side reference ops.
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
_FP8_MIN = float(torch.finfo(_FP8_DTYPE).min)


# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------


def _skip_if_not_rocm():
    """Skip a test if we are not on a ROCm GPU.

    Called at the top of every test function so that the test is *collected*
    by pytest on any machine (no import-time skip) but silently bypassed on
    non-AMD hardware.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")
    if not _IS_HIP:
        pytest.skip("triton_store_cache kernels are ROCm-only; skipping on CUDA")


# ---------------------------------------------------------------------------
# flashmla helpers
# ---------------------------------------------------------------------------

# ---- Layout constants (must match triton_store_cache.py and store.cuh) ----
_NOPE_DIM = 448        # number of nope (non-positional-encoding) dims
_ROPE_DIM = 64         # number of rope (rotary position encoding) dims
_INPUT_DIM = _NOPE_DIM + _ROPE_DIM  # 512 — total kv-head dimension
_TILE_SIZE = 64        # FP8 quantisation tile size (= warp width)
_NUM_NOPE_TILES = _NOPE_DIM // _TILE_SIZE  # 7 — one UE8M0 scale byte per tile
_SCALE_BYTES_PER_TOKEN = 8  # 7 UE8M0 scale bytes + 1 padding byte


def _flashmla_bytes_per_page(page_size: int) -> int:
    """Return the padded byte count per page for the flashmla KV cache.

    Matches the formula in deepseekv4_memory_pool.py:
        bytes_per_page = ceil(584 * page_size / 576) * 576

    The 584 comes from 576 (data bytes per slot) + 8 (scale bytes per slot).
    The result is rounded up to a multiple of 576 to keep alignment.
    """
    import math

    return math.ceil(584 * page_size / 576) * 576


def _make_flashmla_cache(num_pages: int, page_size: int) -> torch.Tensor:
    """Allocate a zeroed flashmla KV cache buffer on CUDA.

    Shape: [num_pages, bytes_per_page] uint8.
    Zeros ensure that untouched slots do not alias stale data.
    """
    bpp = _flashmla_bytes_per_page(page_size)
    return torch.zeros(num_pages, bpp, dtype=torch.uint8, device="cuda")


def _read_flashmla_token(
    cache: torch.Tensor,
    token_idx: int,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract the stored data for one token from the flashmla cache.

    Returns:
        nope_fp8_f32  [448] float32 — FP8 nope values upcast to float32
        rope_bf16     [64]  bfloat16 — rope values (no quantisation)
        scale_ue8m0   [7]   uint8 — one UE8M0 exponent byte per nope tile
    """
    bpp = cache.shape[1]            # bytes per page
    page = token_idx // page_size   # which page this token lives in
    slot = token_idx % page_size    # which slot within the page

    # Flatten to 1-D for simple byte-offset indexing.
    flat = cache.reshape(-1)

    # ---- FP8 nope values ----
    # Data region layout: each slot occupies 576 bytes.
    #   bytes [0,   448) → 448 FP8 values (nope region)
    #   bytes [448, 576) → 64 BF16 values (rope region, 64 × 2 bytes = 128 bytes)
    nope_start = page * bpp + slot * 576
    nope_fp8 = flat[nope_start : nope_start + _NOPE_DIM].view(_FP8_DTYPE).float()

    # ---- BF16 rope values ----
    # Rope starts at byte 448 within the slot's 576-byte data block.
    rope_start = page * bpp + slot * 576 + _NOPE_DIM
    rope_bf16 = flat[rope_start : rope_start + _ROPE_DIM * 2].view(torch.bfloat16)

    # ---- UE8M0 scale bytes ----
    # Scale region starts at byte page_size * 576 within the page (after data region).
    # Each slot gets 8 bytes: 7 UE8M0 scale bytes + 1 pad byte.
    s_offset = page * bpp + page_size * 576 + slot * 8
    scale_u8 = flat[s_offset : s_offset + _NUM_NOPE_TILES].clone()

    return nope_fp8, rope_bf16, scale_u8


def _ue8m0_to_scale(ue8m0_bytes: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 bytes to the *encoded scale* used at dequantisation time.

    The stored byte encodes a biased power-of-2 exponent:
        ue8m0_byte = ceil(log2(scale_raw)) + 127

    Decoding:
        ceil_log2     = ue8m0_byte - 127
        scale_encoded = 2^(ceil_log2) = 2^(ue8m0_byte - 127)

    Quant relationship (kernel side):
        inv_scale = 1 / scale_encoded = 2^(127 - ue8m0_byte)
        fp8       = round(x * inv_scale)

    Dequant relationship (this helper):
        x ≈ fp8 * scale_encoded

    NOTE: the CUDA `inv_scale_ue8m0(exp) = 2^(127 - exp)` macro returns the
    quant-time *inv_scale*, not the dequant-time scale.  Passing that value
    here would over-scale by a factor of `scale_encoded^(-2)`.
    """
    return torch.exp2(ue8m0_bytes.float() - 127.0)


def _dequant_flashmla_token(
    nope_fp8_f32: torch.Tensor,   # [448] float32 — FP8 values already upcast
    scale_u8: torch.Tensor,       # [7]   uint8  — UE8M0 exponent bytes
) -> torch.Tensor:
    """Dequantise FP8 nope values back to float32.

    Each of the 7 tiles (64 elements each) has its own scale.
    We broadcast scale over each tile's 64 elements using repeat_interleave.

    Returns [448] float32 — the reconstructed approximate values.
    """
    scales = _ue8m0_to_scale(scale_u8)                       # [7] float32
    scales_expanded = scales.repeat_interleave(_TILE_SIZE)   # [448] float32
    return nope_fp8_f32 * scales_expanded


# ---------------------------------------------------------------------------
# indexer helpers
# ---------------------------------------------------------------------------

_INDEXER_DIM = 128                  # C4 compressed indexer dimension
_INDEXER_BYTES_PER_TOKEN_DATA = 128  # 128 FP8 bytes per token
_INDEXER_BYTES_PER_TOKEN_SCALE = 4   # 1 FP32 scale per token = 4 bytes


def _indexer_bytes_per_page(page_size: int) -> int:
    """Return the byte count per page for the C4 indexer KV cache.

    Layout: 128 data bytes + 4 scale bytes per token → 132 bytes per token.
    """
    return 132 * page_size


def _make_indexer_cache(num_pages: int, page_size: int) -> torch.Tensor:
    """Allocate a zeroed indexer KV cache buffer on CUDA."""
    return torch.zeros(
        num_pages, _indexer_bytes_per_page(page_size), dtype=torch.uint8, device="cuda"
    )


def _read_indexer_token(
    cache: torch.Tensor,
    token_idx: int,
    page_size: int,
) -> Tuple[torch.Tensor, float]:
    """Extract the stored data for one token from the indexer cache.

    Returns:
        fp8_f32  [128] float32 — FP8 values upcast to float32
        scale    float — the FP32 quantisation scale for this token
    """
    bpp = cache.shape[1]
    page = token_idx // page_size
    slot = token_idx % page_size

    flat = cache.reshape(-1)

    # ---- FP8 values ----
    # Data region: page_size slots × 128 bytes each.
    fp8_start = page * bpp + slot * 128
    fp8_f32 = flat[fp8_start : fp8_start + 128].view(_FP8_DTYPE).float()

    # ---- FP32 scale ----
    # Scale region starts at byte page_size * 128 within the page.
    # Each slot gets exactly 4 bytes (one float32).
    scale_start = page * bpp + 128 * page_size + slot * 4
    scale = flat[scale_start : scale_start + 4].view(torch.float32).item()

    return fp8_f32, scale


def _ref_indexer_store(
    input_bf16: torch.Tensor,   # [N, 128] BF16 input on GPU
    loc: torch.Tensor,          # [N] int64 slot indices on GPU
    num_pages: int,
    page_size: int,
) -> torch.Tensor:
    """Pure-Python reference implementation of the indexer store.

    Quantisation algorithm (must match triton_store_cache.py exactly):
        scale     = max(1e-8, abs_max) / FP8_MAX
        inv_scale = 1.0 / scale
        fp8_vals  = clamp(x_f32 * inv_scale, FP8_MIN, FP8_MAX).to(FP8_DTYPE)

    Runs on CPU to avoid any Triton-vs-reference GPU arithmetic discrepancies.
    Returns the reference cache tensor (on CPU) for byte-level comparison.
    """
    N = input_bf16.shape[0]
    cache = _make_indexer_cache(num_pages, page_size)  # GPU zeros
    flat = cache.reshape(-1)

    # Move to CPU for deterministic Python-level arithmetic.
    x_f32 = input_bf16.float().cpu()
    loc_cpu = loc.cpu()

    for i in range(N):
        idx = int(loc_cpu[i].item())
        page = idx // page_size
        slot = idx % page_size

        row = x_f32[i]
        abs_max = row.abs().max().item()

        # Standard per-token symmetric quantisation.
        scale = max(1e-8, abs_max) / _FP8_MAX
        inv_scale = 1.0 / scale
        fp8_f32 = (row * inv_scale).clamp(_FP8_MIN, _FP8_MAX)
        fp8_vals = fp8_f32.to(_FP8_DTYPE)

        # Write FP8 values as raw bytes into the data region.
        fp8_start = page * _indexer_bytes_per_page(page_size) + slot * 128
        flat[fp8_start : fp8_start + 128] = fp8_vals.view(torch.uint8).cpu()

        # Write the FP32 scale as raw bytes into the scale region.
        scale_start = (
            page * _indexer_bytes_per_page(page_size) + 128 * page_size + slot * 4
        )
        scale_t = torch.tensor([scale], dtype=torch.float32)
        flat[scale_start : scale_start + 4] = scale_t.view(torch.uint8)

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

    We run both paths on the same random input and compare:
      - rope: exactly equal (no quantisation involved)
      - UE8M0 scales: within ±1 exponent unit (adjacent rounding choices)
      - dequantised nope: within FP8-appropriate tolerances (rtol=0.15, atol=0.5)

    Parametrised over page sizes (64/128/256) and token batch sizes with
    non-zero base_index to exercise cross-page scatter addressing.
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

    # Random BF16 input: [N, 512] with values drawn from N(0, 1).
    key = torch.randn((num_tokens, _INPUT_DIM), device=device, dtype=torch.bfloat16)

    # Random slot indices starting at base_index to stress non-zero page offsets.
    # We sample without replacement from a 2× larger pool to avoid slot collisions.
    loc = (
        base_index
        + torch.randperm(num_tokens * 2, device=device)[:num_tokens]
    ).to(torch.int64)

    # Allocate enough pages to hold the highest index.
    num_pages = int(loc.max().item()) // page_size + 2
    bpp = _flashmla_bytes_per_page(page_size)

    # ---- Reference: run the existing two-step AMD fallback ----
    ref_cache = _make_flashmla_cache(num_pages, page_size)
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(key)  # step 1: quant + pack
    _set_k_and_s_triton(ref_cache, loc, pack, page_size)  # step 2: scatter
    torch.cuda.synchronize()

    # ---- Fused kernel ----
    fused_cache = _make_flashmla_cache(num_pages, page_size)
    triton_fused_store_flashmla(key, fused_cache, loc, page_size)
    torch.cuda.synchronize()

    # ---- Per-token comparison ----
    for i in range(num_tokens):
        idx = int(loc[i].item())

        ref_nope, ref_rope, ref_scales = _read_flashmla_token(ref_cache, idx, page_size)
        fus_nope, fus_rope, fus_scales = _read_flashmla_token(fused_cache, idx, page_size)

        # Rope is copied verbatim (no quantisation) — must be bitwise equal.
        torch.testing.assert_close(fus_rope, ref_rope, rtol=0, atol=0)

        # UE8M0 scales: both kernels compute ceil(log2(scale)) independently.
        # A ±1 difference means one kernel chose the next higher power of 2,
        # which is a valid rounding choice and causes at most 2× scale difference.
        scale_diff = (fus_scales.int() - ref_scales.int()).abs()
        assert scale_diff.max().item() <= 1, (
            f"token {i}: UE8M0 scale differs by more than 1 exponent unit: {scale_diff}"
        )

        # Dequantised nope: compare reconstructed float32 values.
        # We use the kernel's own scale for dequantisation, so scale errors do
        # not accumulate on top of quantisation errors.
        ref_deq = _dequant_flashmla_token(ref_nope, ref_scales)
        fus_deq = _dequant_flashmla_token(fus_nope, fus_scales)
        torch.testing.assert_close(fus_deq, ref_deq, rtol=0.15, atol=0.5)


@pytest.mark.parametrize("page_size", [64, 256])
@pytest.mark.parametrize("num_tokens", [1, 64, 257])
def test_flashmla_roundtrip_reconstruction(num_tokens: int, page_size: int):
    """Dequantised output must approximately reconstruct the original BF16 input.

    This is a standalone correctness sanity check that does not depend on the
    two-step fallback being available.  We store the input, read it back, and
    verify that quantisation error is within FP8-level expectations.

    Tolerances: rtol=0.15, atol=5e-2 (looser than the inter-kernel comparison
    because we are measuring absolute quantisation error, not relative drift
    between two equivalent implementations).
    """
    _skip_if_not_rocm()

    from sglang.jit_kernel.triton_store_cache import triton_fused_store_flashmla

    device = torch.device("cuda")
    key = torch.randn((num_tokens, _INPUT_DIM), device=device, dtype=torch.bfloat16)

    # Sequential slot indices for simplicity (one token per slot, starting at 0).
    loc = torch.arange(num_tokens, device=device, dtype=torch.int64)
    num_pages = num_tokens // page_size + 2

    cache = _make_flashmla_cache(num_pages, page_size)
    triton_fused_store_flashmla(key, cache, loc, page_size)
    torch.cuda.synchronize()

    orig_f32 = key.float()

    for i in range(num_tokens):
        nope_fp8, rope_bf16, scales = _read_flashmla_token(cache, i, page_size)

        # Nope: dequantise and compare against the original BF16 values.
        deq_nope = _dequant_flashmla_token(nope_fp8, scales)
        torch.testing.assert_close(
            deq_nope, orig_f32[i, :_NOPE_DIM], rtol=0.15, atol=5e-2
        )
        # Rope: must be exact (lossless BF16 copy, no quantisation).
        torch.testing.assert_close(
            rope_bf16.float(), orig_f32[i, _NOPE_DIM:], rtol=0, atol=0
        )


def test_flashmla_zero_input(page_size: int = 256):
    """Zero input produces zero FP8 nope, zero rope, and a valid (small) scale.

    This guards against divide-by-zero in the scale computation.  The EPS floor
    (1e-8) ensures scale > 0, and clamping ensures the FP8 output is 0.0 when
    the input is 0.0.
    """
    _skip_if_not_rocm()

    from sglang.jit_kernel.triton_store_cache import triton_fused_store_flashmla

    device = torch.device("cuda")
    # Use 4 tokens so we verify multiple slots in the same page.
    key = torch.zeros((4, _INPUT_DIM), device=device, dtype=torch.bfloat16)
    loc = torch.arange(4, device=device, dtype=torch.int64)
    num_pages = 1

    cache = _make_flashmla_cache(num_pages, page_size)
    triton_fused_store_flashmla(key, cache, loc, page_size)
    torch.cuda.synchronize()

    for i in range(4):
        nope_fp8, rope_bf16, _ = _read_flashmla_token(cache, i, page_size)
        # All nope FP8 values must be exactly zero.
        assert (nope_fp8 == 0).all(), f"token {i}: expected zero nope FP8 output"
        # All rope BF16 values must be exactly zero.
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

    The reference (_ref_indexer_store) runs on CPU with explicit Python loops
    so there is no risk of the same Triton bug appearing in both paths.

    Comparison:
      - FP32 scale: near-exact (rtol=1e-4, atol=1e-7 — same float32 arithmetic)
      - Dequantised FP8 values: within FP8-appropriate tolerances
    """
    _skip_if_not_rocm()

    from sglang.jit_kernel.triton_store_cache import triton_fused_store_indexer

    device = torch.device("cuda")
    key = torch.randn(
        (num_tokens, _INDEXER_DIM), device=device, dtype=torch.bfloat16
    )

    # Random slot indices with optional non-zero base to stress page addressing.
    loc = (
        base_index
        + torch.randperm(num_tokens * 2, device=device)[:num_tokens]
    ).to(torch.int64)

    num_pages = int(loc.max().item()) // page_size + 2

    # ---- Fused Triton kernel ----
    fused_cache = _make_indexer_cache(num_pages, page_size)
    triton_fused_store_indexer(key, fused_cache, loc, page_size)
    torch.cuda.synchronize()

    # ---- Pure-Python reference ----
    ref_cache = _ref_indexer_store(key, loc, num_pages, page_size)

    # ---- Per-token comparison ----
    for i in range(num_tokens):
        idx = int(loc[i].item())
        ref_fp8, ref_scale = _read_indexer_token(ref_cache, idx, page_size)
        fus_fp8, fus_scale = _read_indexer_token(fused_cache, idx, page_size)

        # Scale should be essentially identical (both compute the same formula).
        torch.testing.assert_close(
            torch.tensor(fus_scale), torch.tensor(ref_scale), rtol=1e-4, atol=1e-7
        )

        # Dequantise both sides and compare.
        # Multiplying fp8 by scale reconstructs the approximate original values.
        ref_deq = ref_fp8 * ref_scale
        fus_deq = fus_fp8 * fus_scale
        torch.testing.assert_close(fus_deq, ref_deq, rtol=0.15, atol=0.5)


@pytest.mark.parametrize("num_tokens", [1, 64, 257])
def test_indexer_roundtrip_reconstruction(num_tokens: int, page_size: int = 64):
    """Dequantised indexer output must approximately reconstruct the original input.

    Similar to the flashmla roundtrip test but for the C4 indexer path.
    Uses sequential slot indices for simplicity.
    """
    _skip_if_not_rocm()

    from sglang.jit_kernel.triton_store_cache import triton_fused_store_indexer

    device = torch.device("cuda")
    key = torch.randn(
        (num_tokens, _INDEXER_DIM), device=device, dtype=torch.bfloat16
    )
    loc = torch.arange(num_tokens, device=device, dtype=torch.int64)
    num_pages = num_tokens // page_size + 2

    cache = _make_indexer_cache(num_pages, page_size)
    triton_fused_store_indexer(key, cache, loc, page_size)
    torch.cuda.synchronize()

    orig_f32 = key.float()
    for i in range(num_tokens):
        fp8, scale = _read_indexer_token(cache, i, page_size)
        # Dequantise: fp8_value * scale ≈ original.
        deq = fp8 * scale
        torch.testing.assert_close(deq, orig_f32[i], rtol=0.15, atol=5e-2)


def test_indexer_zero_input(page_size: int = 64):
    """Zero input produces zero FP8 values (EPS prevents divide-by-zero in scale).

    The scale will be EPS / FP8_MAX (a very small number), but clamping
    ensures x * inv_scale = 0 for all-zero input.
    """
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
    # Allow running directly: python test_triton_store_cache.py -v
    sys.exit(pytest.main([__file__, "-v", "-s"]))
