"""FP8 path of the V2 fused norm+RoPE+store torch fallback vs the JIT kernel.

Covers both head-dim variants of ``compress_norm_rope_store`` with ``use_fp4=False``:
  * Indexer  (head_dim=128): RMSNorm + RoPE + 128-pt WHT + FP8 -> indexer cache.
  * FlashMLA (head_dim=512): RMSNorm + RoPE -> FlashMLA paged cache.

The torch fallback is forced on via SGLANG_DSV4_FORCE_TORCH_FALLBACK_KERNELS and its
written cache is decoded back to value space (fp8 bytes * scale) and compared against
the JIT kernel's. The comparison is in value space rather than byte-exact because the
software fp8 e4m3 cast and the per-warp abs-max reduction can land one ULP / one scale
exponent off the hardware kernel at quantization boundaries; what matters for model
accuracy is that the dequantized values agree.

This also pins the _cast_to_ue8m0 ceil behaviour: a plain exponent truncation leaves
every FlashMLA nope-warp scale one exponent low, which inflates the FlashMLA value-space
error ~17x (mean abs 0.0019 -> 0.033) and trips the threshold below.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch

from sglang.jit_kernel.benchmark.bench_activation import register_cuda_ci
from sglang.jit_kernel.dsv4 import CompressorDecodePlan, compress_norm_rope_store
from sglang.srt.environ import envs
from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=30, suite="nightly-kernel-1-gpu", nightly=True)

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

# Value-space tolerances. The correct (ceil) fallback sits well under these; the
# truncating bug pushes the FlashMLA mean rel error past 0.01 and trips the test.
_MEAN_REL_TOL = 0.005
_MAX_ABS_TOL = 1.0


def _flashmla_page_bytes() -> int:
    return math.ceil((_MLA_SLOT_BYTES + 8) * PAGE_SIZE / _MLA_SLOT_BYTES) * _MLA_SLOT_BYTES


def _page_bytes(head_dim: int) -> int:
    return 132 * PAGE_SIZE if head_dim == 128 else _flashmla_page_bytes()


def _run(head_dim: int, num_tokens: int, force_fallback: bool) -> torch.Tensor:
    """Store FP8 norm+rope output for ``num_tokens`` decode events and return the cache."""
    torch.manual_seed(num_tokens + head_dim)
    num_pages = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)

    kv = torch.randn(num_tokens, head_dim, device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    # Decode events fire when seq_len % compress_ratio == 0, so make every token one.
    seq_lens = (
        torch.arange(1, num_tokens + 1, device="cuda", dtype=torch.int64)
        * COMPRESS_RATIO
    )
    req_pool_indices = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    plan = CompressorDecodePlan.generate_legacy(
        COMPRESS_RATIO, req_pool_indices, seq_lens
    )
    out_loc = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    freqs_cis = precompute_freqs_cis(
        ROPE_DIM, int(seq_lens.max().item()) + 1, 0, 10000, 1, 32, 1
    ).to("cuda")
    cache = torch.zeros(
        num_pages, _page_bytes(head_dim), device="cuda", dtype=torch.uint8
    )

    sel = "norm_rope" if force_fallback else ""
    with envs.SGLANG_DSV4_FORCE_TORCH_FALLBACK_KERNELS.override(sel):
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
    return cache


def _ue8m0_inv_scale(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 exponent bytes to inv_scale = 2^(127 - e)."""
    inv_exp = (254 - scale_bytes.to(torch.int32)).clamp(min=0)
    inv = (inv_exp << 23).to(torch.int32).view(torch.float32)
    return torch.where(scale_bytes >= 254, torch.zeros_like(inv), inv)


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

    Returns (nope_values (N, 448), rope_bytes (N, 128)).
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
        rope.append(cache[page, vbase + _MLA_NOPE_BYTES : vbase + _MLA_SLOT_BYTES])
    return torch.stack(nope), torch.stack(rope)


def _assert_value_close(jit: torch.Tensor, fb: torch.Tensor, label: str) -> None:
    err = (jit - fb).abs()
    rel = err / jit.abs().clamp(min=1e-6)
    mean_rel = rel.mean().item()
    max_abs = err.max().item()
    assert mean_rel <= _MEAN_REL_TOL and max_abs <= _MAX_ABS_TOL, (
        f"{label}: fallback value mismatch vs JIT "
        f"(mean_rel={mean_rel:.4g} > {_MEAN_REL_TOL} or max_abs={max_abs:.4g} > {_MAX_ABS_TOL})"
    )


@pytest.mark.parametrize("num_tokens", [1, 16, 96])
def test_indexer_fp8_matches_jit(num_tokens: int) -> None:
    """head_dim=128 indexer FP8 fallback dequantizes to the same values as the JIT kernel."""
    jit = _dequant_indexer(_run(128, num_tokens, force_fallback=False), num_tokens)
    fb = _dequant_indexer(_run(128, num_tokens, force_fallback=True), num_tokens)
    _assert_value_close(jit, fb, "indexer")


@pytest.mark.parametrize("num_tokens", [1, 16, 96])
def test_flashmla_fp8_matches_jit(num_tokens: int) -> None:
    """head_dim=512 FlashMLA FP8 fallback: nope values match (within fp8) and rope is bit-exact."""
    jit_nope, jit_rope = _dequant_flashmla(
        _run(512, num_tokens, force_fallback=False), num_tokens
    )
    fb_nope, fb_rope = _dequant_flashmla(
        _run(512, num_tokens, force_fallback=True), num_tokens
    )
    # The bf16 rope region carries no quantization, so it must be bit-exact.
    torch.testing.assert_close(fb_rope, jit_rope, atol=0, rtol=0)
    _assert_value_close(jit_nope, fb_nope, "flashmla nope")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
