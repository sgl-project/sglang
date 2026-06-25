"""FP8 path of the V2 fused norm+RoPE+store kernel vs an inline torch reference.

Covers both head-dim variants of ``compress_norm_rope_store`` with ``use_fp4=False``:
  * Indexer  (head_dim=128): RMSNorm + RoPE + 128-pt WHT + FP8 -> indexer cache.
  * FlashMLA (head_dim=512): RMSNorm + RoPE -> FlashMLA paged cache.

Like test_fp4_indexer, the test calls the public ``compress_norm_rope_store`` wrapper
once and compares the written cache against an independent inline reference. The wrapper
dispatches to the JIT CUDA kernel on CUDA and to the torch fallback on any non-CUDA
device (e.g. XPU), so the fallback is exercised exactly where it is actually used and
is never tested against itself on CUDA.

The cache is decoded back to value space (fp8 bytes * scale) before comparison rather
than compared byte-exact: the software fp8 e4m3 cast and the per-warp abs-max reduction
can land one ULP / one scale exponent off the hardware kernel at quantization
boundaries; what matters for model accuracy is that the dequantized values agree.

This also pins the _cast_to_ue8m0 ceil behaviour: a plain exponent truncation leaves
every FlashMLA nope-warp scale one exponent low, which inflates the FlashMLA value-space
error ~17x (mean abs 0.0019 -> 0.033) and trips the threshold below.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch

from sglang.jit_kernel.dsv4 import CompressorDecodePlan, compress_norm_rope_store
from sglang.jit_kernel.dsv4.fused_norm_rope_v2_torch import _fwht128
from sglang.jit_kernel.hadamard import hadamard_transform
from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis
from sglang.srt.utils import get_device, is_cuda

_is_cuda = is_cuda()

ROPE_DIM = 64
PAGE_SIZE = 64
COMPRESS_RATIO = 4
NORM_EPS = 1.0e-6

# FlashMLA slot (576 B): fp8 nope[0:448] + bf16 rope[448:576]; 7 UE8M0 scale bytes.
_MLA_NOPE_BYTES = 448
_MLA_SLOT_BYTES = 576
_MLA_SCALE_BYTES = 7
_MLA_WARPS = 7
# Indexer slot (132 B): fp8 value[0:128] + fp32 scale[128:132].
_IDX_VALUE_BYTES = 128
_IDX_SCALE_BYTES = 4

# Value-space tolerances. The reference applies the same fp8 quantization as the
# kernel, so the only residual on CUDA is the hardware-vs-software fp8 cast; on XPU
# the kernel is the torch fallback and the match is exact. The truncating ue8m0 bug
# pushes the FlashMLA mean rel error past 0.01 and trips the test.
_FP8_E4M3_MAX = 448.0
_MEAN_REL_TOL = 0.005
_MAX_ABS_TOL = 1.0


def _flashmla_page_bytes() -> int:
    return (
        math.ceil((_MLA_SLOT_BYTES + 8) * PAGE_SIZE / _MLA_SLOT_BYTES) * _MLA_SLOT_BYTES
    )


def _page_bytes(head_dim: int) -> int:
    return 132 * PAGE_SIZE if head_dim == 128 else _flashmla_page_bytes()


def _run(head_dim: int, num_tokens: int):
    """Store FP8 norm+rope output for ``num_tokens`` decode events.

    Returns ``(cache, kv, norm_weight, freqs_cis, positions)`` so the test can build
    an inline reference for the same inputs.
    """
    torch.manual_seed(num_tokens + head_dim)
    device = get_device()
    num_pages = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)

    kv = torch.randn(num_tokens, head_dim, device=device, dtype=torch.bfloat16)
    norm_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
    # Decode events fire when seq_len % compress_ratio == 0, so make every token one.
    seq_lens = (
        torch.arange(1, num_tokens + 1, device=device, dtype=torch.int64)
        * COMPRESS_RATIO
    )
    req_pool_indices = torch.arange(num_tokens, device=device, dtype=torch.int64)
    plan = CompressorDecodePlan.generate_legacy(
        COMPRESS_RATIO, req_pool_indices, seq_lens
    )
    out_loc = torch.arange(num_tokens, device=device, dtype=torch.int32)
    freqs_cis = precompute_freqs_cis(
        ROPE_DIM, int(seq_lens.max().item()) + 1, 0, 10000, 1, 32, 1
    ).to(device)
    cache = torch.zeros(
        num_pages, _page_bytes(head_dim), device=device, dtype=torch.uint8
    )

    compress_norm_rope_store(
        kv.clone(),
        plan,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
        freq_cis=freqs_cis,
        out_loc=out_loc,
        kvcache=cache,
        page_size=PAGE_SIZE,
        use_fp4=False,
    )
    # Decode RoPE position for token i is its seq_len minus the current compress step.
    positions = seq_lens - COMPRESS_RATIO
    return cache, kv, norm_weight, freqs_cis, positions


def _ref_norm_rope(
    kv: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Inline RMSNorm + RoPE (last ROPE_DIM dims). Returns float (N, head_dim)."""
    x = kv.float()
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(rms + NORM_EPS) * norm_weight.float()

    freqs = torch.view_as_real(freqs_cis).flatten(-2)[positions.long()]
    freq_re, freq_im = freqs[:, 0::2], freqs[:, 1::2]

    nope_dim = kv.shape[-1] - ROPE_DIM
    x_rope = x[:, nope_dim:]
    xr, xi = x_rope[:, 0::2], x_rope[:, 1::2]
    rot_re = xr * freq_re - xi * freq_im
    rot_im = xr * freq_im + xi * freq_re
    rotated = torch.stack([rot_re, rot_im], dim=-1).flatten(-2)
    return torch.cat([x[:, :nope_dim], rotated], dim=-1)


def _hadamard_ref(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Device-agnostic Hadamard reference.

    On CUDA use the JIT kernel; elsewhere (e.g. XPU) use the torch-native 128-pt
    WHT the fallback itself uses, scaled to match the kernel's ``scale`` arg.
    """
    if _is_cuda:
        return hadamard_transform(x.contiguous(), scale=scale)
    return _fwht128(x.float().contiguous()) * scale


def _ue8m0_inv_scale(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 exponent bytes to inv_scale = 2^(127 - e)."""
    inv_exp = (254 - scale_bytes.to(torch.int32)).clamp(min=0)
    inv = (inv_exp << 23).to(torch.int32).view(torch.float32)
    return torch.where(scale_bytes >= 254, torch.zeros_like(inv), inv)


def _ref_quant_indexer(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor fp8 quant + dequant of (N, 128), matching the indexer fallback."""
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = abs_max.clamp(min=1e-4) / _FP8_E4M3_MAX  # fp32 scale, stored verbatim
    fp8 = (x / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return fp8.float() * scale


def _ref_quant_flashmla_nope(x_nope: torch.Tensor) -> torch.Tensor:
    """Per-warp (7x64) UE8M0-scaled fp8 quant + dequant of (N, 448) nope region."""
    n = x_nope.shape[0]
    xw = x_nope.reshape(n, _MLA_WARPS, -1)
    abs_max = xw.abs().amax(dim=-1)
    scale_raw = abs_max.clamp(min=1e-4) / _FP8_E4M3_MAX
    ue8m0 = _ceil_ue8m0(scale_raw)
    inv = _ue8m0_inv_scale(ue8m0).unsqueeze(-1)
    fp8 = (
        (xw * inv).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
    )
    scale = 1.0 / inv.clamp(min=1e-30)
    return (fp8 * scale).reshape(n, -1)


def _ceil_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Encode positive fp32 as UE8M0, ceiling the biased exponent (matches kernel)."""
    bits = x.float().clamp(min=1e-38).view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    return (exp + (mant != 0).to(torch.int32)).clamp(max=0xFF).to(torch.uint8)


def _dequant_indexer(cache: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """head_dim=128: dequantize fp8 value bytes with their fp32 scale -> (N, 128)."""
    vals = []
    for tok in range(num_tokens):
        page, off = tok // PAGE_SIZE, tok % PAGE_SIZE
        vbase = off * _IDX_VALUE_BYTES
        sbase = _IDX_VALUE_BYTES * PAGE_SIZE + off * _IDX_SCALE_BYTES
        fp8 = cache[page, vbase : vbase + _IDX_VALUE_BYTES].view(torch.float8_e4m3fn)
        scale = cache[page, sbase : sbase + _IDX_SCALE_BYTES].view(torch.float32)
        vals.append(fp8.float() * scale)
    return torch.stack(vals)


def _dequant_flashmla(cache: torch.Tensor, num_tokens: int):
    """head_dim=512: dequantize the fp8 nope region (7 warps x UE8M0) and read rope.

    Returns (nope_values (N, 448), rope_values (N, 64) float from stored bf16).
    """
    nope, rope = [], []
    for tok in range(num_tokens):
        page, off = tok // PAGE_SIZE, tok % PAGE_SIZE
        vbase = off * _MLA_SLOT_BYTES
        sbase = _MLA_SLOT_BYTES * PAGE_SIZE + off * 8
        fp8 = (
            cache[page, vbase : vbase + _MLA_NOPE_BYTES]
            .view(torch.float8_e4m3fn)
            .float()
        )
        inv = _ue8m0_inv_scale(cache[page, sbase : sbase + _MLA_SCALE_BYTES])
        scale = 1.0 / inv.clamp(min=1e-30)
        nope.append((fp8.view(_MLA_WARPS, -1) * scale.view(_MLA_WARPS, 1)).reshape(-1))
        rope_bytes = cache[page, vbase + _MLA_NOPE_BYTES : vbase + _MLA_SLOT_BYTES]
        rope.append(rope_bytes.view(torch.bfloat16).float())
    return torch.stack(nope), torch.stack(rope)


def _assert_value_close(ref: torch.Tensor, got: torch.Tensor, label: str) -> None:
    err = (ref - got).abs()
    rel = err / ref.abs().clamp(min=1e-6)
    mean_rel = rel.mean().item()
    max_abs = err.max().item()
    assert mean_rel <= _MEAN_REL_TOL and max_abs <= _MAX_ABS_TOL, (
        f"{label}: cache value mismatch vs reference "
        f"(mean_rel={mean_rel:.4g} > {_MEAN_REL_TOL} or max_abs={max_abs:.4g} > {_MAX_ABS_TOL})"
    )


@pytest.mark.parametrize("num_tokens", [1, 16, 96])
def test_indexer_fp8_matches_ref(num_tokens: int) -> None:
    """head_dim=128 indexer FP8: cache dequantizes to the inline norm+rope+WHT reference.

    The reference applies the same per-tensor fp8 quantization, so only the
    hardware-vs-software fp8 cast remains on CUDA (exact on XPU).
    """
    cache, kv, norm_weight, freqs_cis, positions = _run(128, num_tokens)
    ref = _hadamard_ref(
        _ref_norm_rope(kv, norm_weight, freqs_cis, positions), scale=128**-0.5
    )
    ref = _ref_quant_indexer(ref)
    got = _dequant_indexer(cache, num_tokens)
    _assert_value_close(ref, got, "indexer")


@pytest.mark.parametrize("num_tokens", [1, 16, 96])
def test_flashmla_fp8_matches_ref(num_tokens: int) -> None:
    """head_dim=512 FlashMLA FP8: nope values and bf16 rope match the inline reference.

    The reference applies the same per-warp UE8M0 fp8 quantization to the nope region,
    so only the hardware-vs-software fp8 cast remains on CUDA (exact on XPU). The rope
    region is bf16 with no quantization, so it must match bit-exactly.
    """
    cache, kv, norm_weight, freqs_cis, positions = _run(512, num_tokens)
    ref = _ref_norm_rope(kv, norm_weight, freqs_cis, positions)
    ref_nope = _ref_quant_flashmla_nope(ref[:, :_MLA_NOPE_BYTES])
    ref_rope = ref[:, _MLA_NOPE_BYTES:].to(torch.bfloat16).float()
    got_nope, got_rope = _dequant_flashmla(cache, num_tokens)
    torch.testing.assert_close(got_rope, ref_rope, atol=0, rtol=0)
    _assert_value_close(ref_nope, got_nope, "flashmla nope")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
